import pymysql
from app.config import settings

TABLE_MAP = {
    "legal": "法律法规",
    "tender": "政府招标信息",
    "product": "市场产品信息",
}


def _get_conn():
    return pymysql.connect(
        host=settings.database_url,
        user=settings.database_user,
        password=settings.database_password,
        database=settings.database_db_name,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def list_records(category: str, page: int = 1, page_size: int = 20) -> dict:
    table = TABLE_MAP.get(category)
    if not table:
        raise ValueError(f"未知分类: {category}")
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) as total FROM `{table}`")
            total = cur.fetchone()["total"]
            offset = (page - 1) * page_size
            cur.execute(f"SELECT * FROM `{table}` LIMIT {page_size} OFFSET {offset}")
            rows = cur.fetchall()
    finally:
        conn.close()
    return {"rows": rows, "total": total, "page": page, "page_size": page_size}


def get_record(category: str, record_id: int) -> dict | None:
    table = TABLE_MAP.get(category)
    if not table:
        raise ValueError(f"未知分类: {category}")
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM `{table}` WHERE id = %s", (record_id,))
            row = cur.fetchone()
    finally:
        conn.close()
    return row