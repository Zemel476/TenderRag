# -*- coding: utf-8 -*-
"""
db_util.py


Author: shui-
Date: 2026/5/5 17:10
"""
import pymysql
from pymysql.cursors import DictCursor

from app.config import settings


def get_mysql_connection() -> pymysql.Connection:
    try:
        # 默认使用 DictCursor，让返回结果为字典格式，便于外部处理
        conn = pymysql.connect(
            host=settings.database_url,
            user=settings.database_user,
            password=settings.database_password,
            database=settings.database_db_name,
            charset="utf8mb4",
            cursorclass=DictCursor,
        )
        return conn
    except pymysql.Error as e:
        print(f"MySQL 连接失败: {e}")
        raise


def execute_sql(sql: str, params: tuple = None):
    """
    执行一次查询并返回结果（自动管理连接的打开和关闭）。
    适用于简单的即查即用场景。
    """
    conn = None
    cursor = None
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()
        cursor.execute(sql, params)
        if sql.strip().upper().startswith('SELECT'):
            return cursor.fetchall()
        else:
            conn.commit()
            return cursor.rowcount
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()