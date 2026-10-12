"""
Employee Attendance & Analytics API  (FastAPI + MongoDB)

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB   (a local .env is loaded, real environment variables win)

Layout of this file
  1. configuration & database
  2. time / rule helpers (R1 - R5, R8)
  3. validation types and request models
  4. serialisation helpers (BSON <-> API)
  5. indexes (created at startup)
  6. employees, attendance endpoints
  7. analytics pipelines + endpoints
  8. explain endpoint
"""
import calendar
import json
import os
from contextlib import asynccontextmanager
from datetime import date as Date
from datetime import datetime, time, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Any, Literal, Optional

from bson import SON, json_util
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import AfterValidator, BaseModel, Field, StringConstraints, model_validator
from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

# --------------------------------------------------------------------------- #
# 1. Configuration & database
# --------------------------------------------------------------------------- #
load_dotenv()  # override=False by default: real environment variables win over .env

UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30))
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

# tz_aware=True: PyMongo hands back timezone-aware (UTC) datetimes instead of naive ones.
client = MongoClient(
    os.getenv("MONGO_URI", "mongodb://localhost:27017"),
    tz_aware=True,
    tzinfo=UTC,
    serverSelectionTimeoutMS=5000,
)
db = client[os.getenv("MONGO_DB", "attendance_db")]
employees = db["employees"]
logs = db["attendance_logs"]

PRESENT_STATUSES = ["PRESENT", "WFH", "ON_DUTY"]
StatusT = Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]
GRACE_SECONDS = 10 * 60
MIN_OVERTIME_MINUTES = 30
HALF_DAY_LIMIT = Decimal("4.50")


# --------------------------------------------------------------------------- #
# 2. Time and rule helpers
# --------------------------------------------------------------------------- #
def ms_to_dt(ms: int) -> datetime:
    """Epoch ms -> aware UTC datetime, truncated to whole seconds (R1)."""
    return EPOCH + timedelta(milliseconds=(ms // 1000) * 1000)


def ms_to_dt_exact(ms: int) -> datetime:
    """Epoch ms -> aware UTC datetime WITHOUT second-truncation (for server-side timestamps)."""
    return EPOCH + timedelta(milliseconds=ms)


def now_dt() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def dt_to_ms(dt: Optional[datetime]) -> Optional[int]:
    """Datetime -> epoch ms. Naive datetimes are treated as UTC (that is what Mongo stores)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return (dt - EPOCH) // timedelta(milliseconds=1)


def round_half_up(value, places: int) -> float:
    """Half-up rounding on the decimal representation (R4 / R8). Python's round() is banker's."""
    d = value if isinstance(value, Decimal) else Decimal(str(value))
    return float(d.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def hhmm_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def is_overnight(shift_start: str, shift_end: str) -> bool:
    return hhmm_minutes(shift_end) <= hhmm_minutes(shift_start)


def ist_moment(date_str: str, hhmm: str) -> datetime:
    h, m = hhmm.split(":")
    return datetime.combine(Date.fromisoformat(date_str), time(int(h), int(m)), tzinfo=IST)


def attendance_date(punch_in: datetime, shift_start: str, shift_end: str) -> str:
    """R1: IST calendar date of the punch-in; for overnight shifts an early-morning punch-in
    (earlier than shift_end) belongs to the shift that started the previous day."""
    ist = punch_in.astimezone(IST)
    d = ist.date()
    if is_overnight(shift_start, shift_end):
        seconds_of_day = ist.hour * 3600 + ist.minute * 60 + ist.second
        if seconds_of_day < hhmm_minutes(shift_end) * 60:
            d -= timedelta(days=1)
    return d.isoformat()


def compute_late_minutes(punch_in: datetime, date_str: str, shift_start: str) -> int:
    """R2: late only if STRICTLY more than 10:00 after shift_start; then floor minutes since shift_start."""
    seconds = int((punch_in - ist_moment(date_str, shift_start)).total_seconds())
    return seconds // 60 if seconds > GRACE_SECONDS else 0


def compute_overtime(punch_out: datetime, date_str: str, shift_start: str, shift_end: str) -> int:
    """R3: floor minutes from shift_end to punch_out, only if >= 30. Overnight shifts end the next day."""
    end = ist_moment(date_str, shift_end)
    if is_overnight(shift_start, shift_end):
        end += timedelta(days=1)
    minutes = int((punch_out - end).total_seconds()) // 60
    return minutes if minutes >= MIN_OVERTIME_MINUTES else 0


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    """R4: seconds / 3600 rounded to 2 decimals, half-up."""
    seconds = int((punch_out - punch_in).total_seconds())
    return round_half_up(Decimal(seconds) / Decimal(3600), 2)


def derived_fields(emp: dict, status: str, date_str: str, p_in: Optional[datetime], p_out: Optional[datetime]) -> dict:
    """All four stored derived values for a given final state (see DATA_MODEL.md table)."""
    out = {"work_hours": None, "late_minutes": 0, "overtime_minutes": 0, "half_day": False}
    if status not in PRESENT_STATUSES or p_in is None:
        return out
    out["late_minutes"] = compute_late_minutes(p_in, date_str, emp["shift_start"])
    if p_out is not None:
        wh = compute_work_hours(p_in, p_out)
        out["work_hours"] = wh
        out["overtime_minutes"] = compute_overtime(p_out, date_str, emp["shift_start"], emp["shift_end"])
        out["half_day"] = Decimal(str(wh)) < HALF_DAY_LIMIT
    return out


def month_bounds(month: str) -> tuple[str, str]:
    """'2026-07' -> ('2026-07-01', '2026-07-31')."""
    y, m = int(month[:4]), int(month[5:])
    return f"{y:04d}-{m:02d}-01", f"{y:04d}-{m:02d}-{calendar.monthrange(y, m)[1]:02d}"


def count_working_days(first: str, last: str) -> int:
    """R7: Mon-Fri days in [first, last] (a handful of calendar days, not data processing)."""
    d, end, n = Date.fromisoformat(first), Date.fromisoformat(last), 0
    while d <= end:
        n += d.weekday() < 5
        d += timedelta(days=1)
    return n


# --------------------------------------------------------------------------- #
# 3. Validation types and request models
# --------------------------------------------------------------------------- #
def _check_date(v: str) -> str:
    try:
        Date.fromisoformat(v)
    except ValueError:
        raise ValueError("must be a valid calendar date YYYY-MM-DD")
    return v


def _check_month(v: str) -> str:
    if int(v[:4]) < 1:
        raise ValueError("year out of range")
    return v


DateStr = Annotated[str, StringConstraints(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"), AfterValidator(_check_date)]
MonthStr = Annotated[str, StringConstraints(pattern=r"^[0-9]{4}-(0[1-9]|1[0-2])$"), AfterValidator(_check_month)]
HHMM = r"^([01][0-9]|2[0-3]):[0-5][0-9]$"
# strict=True: only real JSON integers. Floats ("1.5e12"), numeric strings and booleans are rejected (422).
EpochMs = Annotated[int, Field(strict=True, ge=100_000_000_000, le=4_102_444_800_000)]
Page = Annotated[int, Query(ge=1)]
PageSize = Annotated[int, Query(ge=1, le=100)]


class EmployeeIn(BaseModel):
    emp_code: str = Field(pattern=r"^EMP[0-9]{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = Field("09:30", pattern=HHMM)
    shift_end: str = Field("18:30", pattern=HHMM)
    joined_on: DateStr

    @model_validator(mode="after")
    def _shifts_differ(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        return self


class PunchInIn(BaseModel):
    emp_code: str
    punched_at: Optional[EpochMs] = None
    status: Literal["PRESENT", "WFH", "ON_DUTY"] = "PRESENT"


class PunchOutIn(BaseModel):
    emp_code: str
    punched_at: Optional[EpochMs] = None


class RegularizeIn(BaseModel):
    status: Optional[StatusT] = None
    punch_in: Optional[EpochMs] = None
    punch_out: Optional[EpochMs] = None
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)


# --------------------------------------------------------------------------- #
# 4. Serialisation helpers (BSON -> API)
# --------------------------------------------------------------------------- #
def employee_out(doc: dict) -> dict:
    return {
        "emp_code": doc["emp_code"],
        "name": doc["name"],
        "email": doc["email"],
        "department": doc["department"],
        "shift_start": doc["shift_start"],
        "shift_end": doc["shift_end"],
        "joined_on": doc["joined_on"],
        "created_at": dt_to_ms(doc.get("created_at")),
    }


def _history_value(v):
    return dt_to_ms(v) if isinstance(v, datetime) else v


def record_out(doc: dict) -> dict:
    """Stored attendance document -> API shape. Missing legacy fields get their defaults; _id never leaves."""
    history = []
    for h in doc.get("history") or []:
        changes = {
            field: {"from": _history_value(c.get("from")), "to": _history_value(c.get("to"))}
            for field, c in (h.get("changes") or {}).items()
        }
        history.append({"at": dt_to_ms(h.get("at")), "by": h.get("by"), "reason": h.get("reason"), "changes": changes})
    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": dt_to_ms(doc.get("punch_in")),
        "punch_out": dt_to_ms(doc.get("punch_out")),
        "work_hours": doc.get("work_hours"),
        "late_minutes": doc.get("late_minutes") or 0,
        "overtime_minutes": doc.get("overtime_minutes") or 0,
        "half_day": bool(doc.get("half_day")),
        "history": history,
    }


def get_employee_or_404(emp_code: str) -> dict:
    emp = employees.find_one({"emp_code": emp_code})
    if emp is None:
        raise HTTPException(404, "employee not found")
    return emp


# --------------------------------------------------------------------------- #
# 5. Indexes (idempotent, created at startup)
# --------------------------------------------------------------------------- #
def ensure_indexes() -> None:
    # employees
    employees.create_index([("emp_code", ASCENDING)], unique=True, name="uq_emp_code")
    employees.create_index([("department", ASCENDING), ("emp_code", ASCENDING)], name="ix_dept_emp")
    employees.create_index([("department", ASCENDING), ("joined_on", ASCENDING)], name="ix_dept_joined")
    employees.create_index([("joined_on", ASCENDING)], name="ix_joined")
    # attendance_logs
    logs.create_index([("emp_code", ASCENDING), ("date", ASCENDING)], unique=True, name="uq_emp_date")
    logs.create_index([("date", DESCENDING), ("emp_code", ASCENDING)], name="ix_date_emp")
    logs.create_index([("status", ASCENDING), ("date", DESCENDING), ("emp_code", ASCENDING)], name="ix_status_date_emp")
    logs.create_index([("emp_code", ASCENDING), ("punch_in", DESCENDING)], name="ix_emp_punchin")
    logs.create_index(
        [("date", ASCENDING), ("late_minutes", ASCENDING), ("emp_code", ASCENDING)], name="ix_late_cover"
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_indexes()
    yield


app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# 6a. Health & employees
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    try:
        client.admin.command("ping")
    except PyMongoError:
        raise HTTPException(503, "database unavailable")
    return {"status": "ok"}


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    doc = body.model_dump()
    doc["created_at"] = ms_to_dt_exact(dt_to_ms(datetime.now(UTC)))  # whole ms, same precision Mongo stores
    try:
        employees.insert_one(doc)  # the unique index on emp_code is the guarantee, not a prior find
    except DuplicateKeyError:
        raise HTTPException(409, "emp_code already exists")
    return employee_out(doc)


@app.get("/employees")
def list_employees(department: Optional[str] = None, page: Page = 1, page_size: PageSize = 20):
    q = {"department": department} if department else {}
    total = employees.count_documents(q)
    cursor = employees.find(q).sort("emp_code", ASCENDING).skip((page - 1) * page_size).limit(page_size)
    return {"items": [employee_out(d) for d in cursor], "total": total, "page": page, "page_size": page_size}


# --------------------------------------------------------------------------- #
# 6b. Attendance
# --------------------------------------------------------------------------- #
@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    emp = get_employee_or_404(body.emp_code)
    ts = ms_to_dt(body.punched_at) if body.punched_at is not None else now_dt()
    d = attendance_date(ts, emp["shift_start"], emp["shift_end"])
    doc = {
        "emp_code": body.emp_code,
        "date": d,
        "status": body.status,
        "punch_in": ts,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": compute_late_minutes(ts, d, emp["shift_start"]),
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }
    try:
        logs.insert_one(doc)  # unique (emp_code, date): of N parallel requests exactly one wins
    except DuplicateKeyError:
        raise HTTPException(409, "already punched in for this date")
    return record_out(doc)


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutIn):
    emp = get_employee_or_404(body.emp_code)
    ts = ms_to_dt(body.punched_at) if body.punched_at is not None else now_dt()
    rec = logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$lte": ts}},
        sort=[("punch_in", DESCENDING)],
    )
    if rec is None:
        raise HTTPException(404, "no punch-in found")
    if rec.get("punch_out") is not None:
        raise HTTPException(409, "already punched out")
    p_in = rec["punch_in"]
    if ts <= p_in:
        raise HTTPException(422, "punched_at must be after punch_in")
    if ts - p_in > timedelta(hours=24):
        raise HTTPException(422, "punch_out is more than 24 hours after punch_in")

    fields = derived_fields(emp, rec["status"], rec["date"], p_in, ts)
    fields["punch_out"] = ts
    fields.pop("late_minutes")  # late_minutes is a punch-in value; it is not touched here
    # Compare-and-set: only succeeds while punch_out is still empty, so two simultaneous
    # punch-outs give exactly one 200.
    updated = logs.find_one_and_update(
        {"_id": rec["_id"], "punch_out": None},
        {"$set": fields},
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(409, "already punched out")
    return record_out(updated)


def build_attendance_query(emp_code, date_from, date_to, status, page, page_size):
    """Shared by GET /attendance and the explain endpoint so both run the same query."""
    if date_from and date_to and date_from > date_to:
        raise HTTPException(422, "date_from must not be after date_to")
    q: dict[str, Any] = {}
    if emp_code:
        q["emp_code"] = emp_code
    if date_from or date_to:
        q["date"] = {}
        if date_from:
            q["date"]["$gte"] = date_from
        if date_to:
            q["date"]["$lte"] = date_to
    if status:
        q["status"] = status
    sort = [("date", DESCENDING), ("emp_code", ASCENDING)]
    return q, sort, (page - 1) * page_size, page_size


@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[DateStr] = None,
    date_to: Optional[DateStr] = None,
    status: Optional[StatusT] = None,
    page: Page = 1,
    page_size: PageSize = 20,
):
    q, sort, skip, limit = build_attendance_query(emp_code, date_from, date_to, status, page, page_size)
    total = logs.count_documents(q) if q else logs.estimated_document_count()
    items = [record_out(d) for d in logs.find(q).sort(sort).skip(skip).limit(limit)]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@app.patch("/attendance/{emp_code}/{date}")
def regularize(emp_code: str, date: Annotated[DateStr, Path()], body: RegularizeIn):
    emp = get_employee_or_404(emp_code)
    for _attempt in range(5):  # optimistic concurrency: retry if someone else changed the record
        rec = logs.find_one({"emp_code": emp_code, "date": date})
        if rec is None:
            raise HTTPException(404, "no attendance record for that date")

        status = body.status or rec["status"]
        if status in PRESENT_STATUSES:
            new_in = ms_to_dt(body.punch_in) if body.punch_in is not None else rec.get("punch_in")
            new_out = ms_to_dt(body.punch_out) if body.punch_out is not None else rec.get("punch_out")
            if new_in is None:
                raise HTTPException(422, "a presence status requires a punch_in")
            if body.punch_in is not None and attendance_date(new_in, emp["shift_start"], emp["shift_end"]) != date:
                raise HTTPException(422, "punch_in must stay on the record's attendance date")
            if new_out is not None and not (new_in < new_out <= new_in + timedelta(hours=24)):
                raise HTTPException(422, "punch_out must be after punch_in and within 24 hours")
        else:  # ABSENT / LEAVE clear both punch times
            if body.punch_in is not None or body.punch_out is not None:
                raise HTTPException(422, "ABSENT/LEAVE cannot have punch times")
            new_in = new_out = None

        new = {"status": status, "punch_in": new_in, "punch_out": new_out}
        new.update(derived_fields(emp, status, date, new_in, new_out))
        old = {
            "status": rec["status"],
            "punch_in": rec.get("punch_in"),
            "punch_out": rec.get("punch_out"),
            "work_hours": rec.get("work_hours"),
            "late_minutes": rec.get("late_minutes") or 0,
            "overtime_minutes": rec.get("overtime_minutes") or 0,
            "half_day": bool(rec.get("half_day")),
        }
        changes = {k: {"from": old[k], "to": new[k]} for k in new if new[k] != old[k]}
        if not changes:
            raise HTTPException(422, "request changes nothing")

        entry = {"at": now_dt(), "by": body.regularized_by, "reason": body.reason, "changes": changes}
        n_hist = len(rec.get("history") or [])
        # Only apply if the record is still exactly what we read (state + history length).
        updated = logs.find_one_and_update(
            {
                "_id": rec["_id"],
                "status": old["status"],
                "punch_in": old["punch_in"],
                "punch_out": old["punch_out"],
                "$expr": {"$eq": [{"$size": {"$ifNull": ["$history", []]}}, n_hist]},
            },
            {"$set": new, "$push": {"history": entry}},
            return_document=ReturnDocument.AFTER,
        )
        if updated is not None:
            return record_out(updated)
    raise HTTPException(409, "concurrent modification, please retry")


# --------------------------------------------------------------------------- #
# 7. Analytics: aggregation pipelines (built by functions so explain can reuse them)
# --------------------------------------------------------------------------- #
# Expressions over one attendance document. Stored derived values are read, never recomputed.
IS_PRESENT = {"$in": ["$status", PRESENT_STATUSES]}
IS_WEEKDAY = {"$lte": [{"$isoDayOfWeek": {"$dateFromString": {"dateString": "$date", "format": "%Y-%m-%d"}}}, 5]}
PRESENT_VALUE = {  # R5-R7: Mon-Fri presence = 1, or 0.5 for a half day
    "$cond": [{"$and": [IS_PRESENT, IS_WEEKDAY]}, {"$cond": [{"$ifNull": ["$half_day", False]}, 0.5, 1]}, 0]
}
LATE_MIN = {"$ifNull": ["$late_minutes", 0]}
IS_LATE = {"$gt": [LATE_MIN, 0]}
HAS_HOURS = {"$and": [IS_PRESENT, {"$in": [{"$type": "$work_hours"}, ["double", "int", "long", "decimal"]]}]}


def _count_if(cond) -> dict:
    return {"$sum": {"$cond": [cond, 1, 0]}}


def _clean_sum(x) -> Decimal:
    """Sums of doubles can carry float noise (18.349999999); snap to 6 decimals before dividing."""
    return Decimal(str(round(x or 0, 6)))


def employee_monthly_pipeline(emp_code: str, month: str) -> list:
    first, last = month_bounds(month)
    return [
        {"$match": {"emp_code": emp_code, "date": {"$gte": first, "$lte": last}}},
        {
            "$group": {
                "_id": None,
                "present_days": {"$sum": PRESENT_VALUE},
                "leave_days": _count_if({"$eq": ["$status", "LEAVE"]}),
                "late_count": _count_if(IS_LATE),
                "total_late_minutes": {"$sum": LATE_MIN},
                "total_overtime_minutes": {"$sum": {"$ifNull": ["$overtime_minutes", 0]}},
            }
        },
    ]


@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: Annotated[MonthStr, Query()]):
    emp = get_employee_or_404(emp_code)
    first, last = month_bounds(month)
    working_days = count_working_days(max(first, emp["joined_on"]), last) if emp["joined_on"] <= last else 0
    row = next(iter(logs.aggregate(employee_monthly_pipeline(emp_code, month))), None) or {}
    present = float(row.get("present_days", 0))
    pct = round_half_up(Decimal(str(present)) / Decimal(working_days) * 100, 2) if working_days else None
    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": present,
        "leave_days": int(row.get("leave_days", 0)),
        "late_count": int(row.get("late_count", 0)),
        "total_late_minutes": int(row.get("total_late_minutes", 0)),
        "total_overtime_minutes": int(row.get("total_overtime_minutes", 0)),
        "attendance_pct": pct,
    }


def department_summary_pipeline(month: str, department: Optional[str]) -> tuple[list, str]:
    """Starts from `employees` so people with zero logs are still counted (R9), then joins their
    month of logs. The per-employee stats are computed INSIDE the $lookup so no big arrays are built."""
    first, last = month_bounds(month)
    match: dict[str, Any] = {"joined_on": {"$lte": last}}
    if department:
        match["department"] = department
    per_employee = [
        {"$match": {"date": {"$gte": first, "$lte": last}}},
        {
            "$group": {
                "_id": None,
                "present_days": {"$sum": PRESENT_VALUE},
                "wh_sum": {"$sum": {"$cond": [HAS_HOURS, "$work_hours", 0]}},
                "wh_cnt": _count_if(HAS_HOURS),
                "late_count": _count_if(IS_LATE),
                "total_late_minutes": {"$sum": LATE_MIN},
                "leave_count": _count_if({"$eq": ["$status", "LEAVE"]}),
                "on_duty_count": _count_if({"$eq": ["$status", "ON_DUTY"]}),
            }
        },
    ]
    pipeline = [
        {"$match": match},
        {
            "$lookup": {
                "from": "attendance_logs",
                "localField": "emp_code",
                "foreignField": "emp_code",
                "pipeline": per_employee,
                "as": "s",
            }
        },
        {"$set": {"s": {"$first": "$s"}}},  # [] -> missing, handled with $ifNull below
        {
            "$group": {
                "_id": "$department",
                "headcount": {"$sum": 1},
                "present_days": {"$sum": {"$ifNull": ["$s.present_days", 0]}},
                "wh_sum": {"$sum": {"$ifNull": ["$s.wh_sum", 0]}},
                "wh_cnt": {"$sum": {"$ifNull": ["$s.wh_cnt", 0]}},
                "late_count": {"$sum": {"$ifNull": ["$s.late_count", 0]}},
                "total_late_minutes": {"$sum": {"$ifNull": ["$s.total_late_minutes", 0]}},
                "leave_count": {"$sum": {"$ifNull": ["$s.leave_count", 0]}},
                "on_duty_count": {"$sum": {"$ifNull": ["$s.on_duty_count", 0]}},
            }
        },
        {"$sort": {"_id": 1}},
    ]
    return pipeline, ("ix_dept_joined" if department else "ix_joined")


@app.get("/analytics/departments/summary")
def department_summary(month: Annotated[MonthStr, Query()], department: Optional[str] = None):
    pipeline, hint = department_summary_pipeline(month, department)
    items = []
    for r in employees.aggregate(pipeline, hint=hint):
        avg = round_half_up(_clean_sum(r["wh_sum"]) / Decimal(r["wh_cnt"]), 2) if r["wh_cnt"] else None
        items.append(
            {
                "department": r["_id"],
                "headcount": r["headcount"],
                "present_days": float(r["present_days"]),
                "avg_work_hours": avg,
                "late_count": r["late_count"],
                "total_late_minutes": r["total_late_minutes"],
                "leave_count": r["leave_count"],
                "on_duty_count": r["on_duty_count"],
            }
        )
    return {"month": month, "items": items}


def late_leaderboard_pipeline(month: str, limit: int, department: Optional[str]) -> tuple[list, str]:
    first, last = month_bounds(month)
    pipeline: list[dict] = [
        {"$match": {"date": {"$gte": first, "$lte": last}, "late_minutes": {"$gt": 0}}},
        {"$group": {"_id": "$emp_code", "total": {"$sum": "$late_minutes"}, "cnt": {"$sum": 1}}},
        {"$lookup": {"from": "employees", "localField": "_id", "foreignField": "emp_code", "as": "e"}},
        {"$unwind": "$e"},  # logs of unknown employees disappear here
    ]
    if department:
        pipeline.append({"$match": {"e.department": department}})  # rank inside the department only
    pipeline += [
        # $rank = standard competition ranking: ties share a rank, the next rank is skipped (1,2,2,4)
        {"$setWindowFields": {"sortBy": {"total": -1}, "output": {"rank": {"$rank": {}}}}},
        {"$match": {"rank": {"$lte": limit}}},  # limit applies AFTER ranking
        {"$sort": {"total": -1, "_id": 1}},
        {
            "$project": {
                "_id": 0,
                "rank": 1,
                "emp_code": "$_id",
                "name": "$e.name",
                "department": "$e.department",
                "total_late_minutes": "$total",
                "late_count": "$cnt",
            }
        },
    ]
    return pipeline, "ix_late_cover"


@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: Annotated[MonthStr, Query()],
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    department: Optional[str] = None,
):
    pipeline, hint = late_leaderboard_pipeline(month, limit, department)
    return {"month": month, "items": list(logs.aggregate(pipeline, hint=hint))}


MAX_TREND_DAYS = 92


def check_trend_range(date_from: str, date_to: str) -> None:
    span = (Date.fromisoformat(date_to) - Date.fromisoformat(date_from)).days + 1
    if span < 1:
        raise HTTPException(422, "'to' must not be before 'from'")
    if span > MAX_TREND_DAYS:
        raise HTTPException(422, f"range must not exceed {MAX_TREND_DAYS} days")


def department_trend_pipeline(department: str, date_from: str, date_to: str) -> tuple[list, str]:
    """Event-style pipeline, all in the database:
       1. every employee emits one 'joined' event (+1 headcount on max(joined_on, from))
          and one event per log in range (present value, late flag), plus a zero 'anchor' on `from`;
       2. group events per day -> sparse daily rows;  3. $densify fills every calendar day;
       4. running $sum of joins = headcount;  5. $setWindowFields 7-row moving average."""
    start = datetime.combine(Date.fromisoformat(date_from), time(), tzinfo=UTC)
    end = datetime.combine(Date.fromisoformat(date_to), time(), tzinfo=UTC)
    log_events = [
        {"$match": {"date": {"$gte": date_from, "$lte": date_to}}},
        {
            "$project": {
                "_id": 0,
                "date": 1,
                "present": {"$cond": [IS_PRESENT, {"$cond": [{"$ifNull": ["$half_day", False]}, 0.5, 1]}, 0]},
                "late": {"$cond": [IS_LATE, 1, 0]},
            }
        },
    ]
    pipeline = [
        {"$match": {"department": department}},
        {
            "$lookup": {
                "from": "attendance_logs",
                "localField": "emp_code",
                "foreignField": "emp_code",
                "pipeline": log_events,
                "as": "logs",
            }
        },
        {
            "$project": {
                "_id": 0,
                "ev": {
                    "$concatArrays": [
                        [
                            {"date": {"$max": ["$joined_on", date_from]}, "hc": {"$cond": [{"$lte": ["$joined_on", date_to]}, 1, 0]}, "present": 0, "late": 0},
                            {"date": date_from, "hc": 0, "present": 0, "late": 0},  # anchor: guarantees >= 1 row for $densify
                        ],
                        {"$map": {"input": "$logs", "as": "l", "in": {"date": "$$l.date", "hc": 0, "present": "$$l.present", "late": "$$l.late"}}},
                    ]
                },
            }
        },
        {"$unwind": "$ev"},
        {"$replaceRoot": {"newRoot": "$ev"}},
        {"$match": {"date": {"$gte": date_from, "$lte": date_to}}},
        {"$group": {"_id": "$date", "hc": {"$sum": "$hc"}, "present": {"$sum": "$present"}, "late": {"$sum": "$late"}}},
        {"$project": {"_id": 0, "d": {"$dateFromString": {"dateString": "$_id", "format": "%Y-%m-%d"}}, "hc": 1, "present": 1, "late": 1}},
        {"$densify": {"field": "d", "range": {"step": 1, "unit": "day", "bounds": [start, end + timedelta(days=1)]}}},
        {"$match": {"d": {"$lte": end}}},
        {"$set": {"hc": {"$ifNull": ["$hc", 0]}, "present": {"$ifNull": ["$present", 0]}, "late": {"$ifNull": ["$late", 0]}}},
        {"$setWindowFields": {"sortBy": {"d": 1}, "output": {"headcount": {"$sum": "$hc", "window": {"documents": ["unbounded", "current"]}}}}},
        {"$set": {"wd": {"$lte": [{"$isoDayOfWeek": "$d"}, 5]}}},
        {
            "$set": {
                "rate": {"$cond": [{"$and": ["$wd", {"$gt": ["$headcount", 0]}]}, {"$divide": ["$present", "$headcount"]}, None]}
            }
        },
        # $avg ignores nulls, so non-working days drop out of the window; all-null window -> null
        {"$setWindowFields": {"sortBy": {"d": 1}, "output": {"ma": {"$avg": "$rate", "window": {"documents": [-6, 0]}}}}},
        {"$sort": {"d": 1}},
        {"$project": {"_id": 0, "date": {"$dateToString": {"date": "$d", "format": "%Y-%m-%d"}}, "wd": 1, "headcount": 1, "present": 1, "late": 1, "ma": 1}},
    ]
    return pipeline, "ix_dept_joined"


@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str,
    date_from: Annotated[DateStr, Query(alias="from")],
    date_to: Annotated[DateStr, Query(alias="to")],
):
    check_trend_range(date_from, date_to)
    if employees.find_one({"department": department}, {"_id": 1}) is None:
        raise HTTPException(404, "unknown department")
    pipeline, hint = department_trend_pipeline(department, date_from, date_to)
    items = []
    for r in employees.aggregate(pipeline, hint=hint):
        hc, present = int(r["headcount"]), r["present"]
        rate = round_half_up(Decimal(str(present)) / Decimal(hc), 4) if (r["wd"] and hc > 0) else None
        ma = round_half_up(round(r["ma"], 10), 4) if r.get("ma") is not None else None
        items.append(
            {
                "date": r["date"],
                "is_working_day": bool(r["wd"]),
                "headcount": hc,
                "present_count": float(present),
                "late_count": int(r["late"]),
                "attendance_rate": rate,
                "moving_avg_7d": ma,
            }
        )
    return {"department": department, "items": items}


# --------------------------------------------------------------------------- #
# 8. Explain endpoint
# --------------------------------------------------------------------------- #
def _require(**params) -> None:
    missing = [k for k, v in params.items() if v is None]
    if missing:
        raise HTTPException(422, f"missing required query parameter(s): {', '.join(missing)}")


@app.get("/admin/explain/{endpoint}")
def explain(
    endpoint: Literal["attendance_list", "employee_monthly", "department_summary", "late_leaderboard", "department_trend"],
    emp_code: Optional[str] = None,
    month: Optional[MonthStr] = None,
    department: Optional[str] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    date_from: Optional[DateStr] = None,
    date_to: Optional[DateStr] = None,
    status: Optional[StatusT] = None,
    from_: Annotated[Optional[DateStr], Query(alias="from")] = None,
    to: Annotated[Optional[DateStr], Query()] = None,
    page: Page = 1,
    page_size: PageSize = 20,
):
    # Each branch builds the command with the SAME builder function the real endpoint uses.
    if endpoint == "attendance_list":
        q, sort, skip, lim = build_attendance_query(emp_code, date_from, date_to, status, page, page_size)
        coll = "attendance_logs"
        inner = {"find": coll, "filter": q, "sort": SON(sort), "skip": skip, "limit": lim}
    else:
        if endpoint == "employee_monthly":
            _require(emp_code=emp_code, month=month)
            coll, pipeline, hint = "attendance_logs", employee_monthly_pipeline(emp_code, month), None
        elif endpoint == "department_summary":
            _require(month=month)
            pipeline, hint = department_summary_pipeline(month, department)
            coll = "employees"
        elif endpoint == "late_leaderboard":
            _require(month=month)
            pipeline, hint = late_leaderboard_pipeline(month, limit, department)
            coll = "attendance_logs"
        else:  # department_trend
            _require(department=department, **{"from": from_, "to": to})
            check_trend_range(from_, to)
            pipeline, hint = department_trend_pipeline(department, from_, to)
            coll = "employees"
        inner = {"aggregate": coll, "pipeline": pipeline, "cursor": {}}
        if hint:
            inner["hint"] = hint
    result = db.command({"explain": inner, "verbosity": "executionStats"})
    return {"endpoint": endpoint, "collection": coll, "explain": json.loads(json_util.dumps(result))}