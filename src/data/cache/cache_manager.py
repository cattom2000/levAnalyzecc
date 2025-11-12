"""
SQLite数据缓存系统
提供高效的数据存储、检索和缓存管理功能
"""

import sqlite3
import pandas as pd
import json
import hashlib
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading
import asyncio
import aiosqlite

from ...contracts.data_sources import ICacheManager, DataQuery, DataSourceType
from ...utils.settings import get_settings
from ...utils.logging import get_logger, handle_errors, ErrorCategory


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    data: Union[pd.DataFrame, Dict[str, Any], bytes]
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    source_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    data_type: str = "dataframe"  # dataframe, dict, binary

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.last_accessed is None:
            self.last_accessed = datetime.now()

    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于数据库存储）"""
        data_dict = asdict(self)
        # 转换datetime对象
        for field in ['created_at', 'expires_at', 'last_accessed']:
            if data_dict[field] is not None:
                data_dict[field] = data_dict[field].isoformat()

        # 序列化数据
        if isinstance(self.data, pd.DataFrame):
            data_dict['data'] = {
                'type': 'dataframe',
                'content': self.data.to_json(orient='records', date_format='iso'),
                'columns': list(self.data.columns),
                'dtypes': self.data.dtypes.to_dict()
            }
        elif isinstance(self.data, dict):
            data_dict['data'] = {
                'type': 'dict',
                'content': json.dumps(self.data)
            }
        else:  # binary data
            data_dict['data'] = {
                'type': 'binary',
                'content': self.data
            }

        # 序列化metadata
        data_dict['metadata'] = json.dumps(self.metadata)
        return data_dict

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> 'CacheEntry':
        """从字典创建缓存条目"""
        # 转换datetime对象
        for field in ['created_at', 'expires_at', 'last_accessed']:
            if data_dict[field] is not None:
                data_dict[field] = datetime.fromisoformat(data_dict[field])

        # 反序列化数据
        data_info = data_dict.pop('data')
        if data_info['type'] == 'dataframe':
            data = pd.read_json(data_info['content'], orient='records')
            # 重建数据类型
            for col, dtype in data_info['dtypes'].items():
                try:
                    data[col] = data[col].astype(dtype)
                except:
                    pass  # 如果类型转换失败，保持原类型
        elif data_info['type'] == 'dict':
            data = json.loads(data_info['content'])
        else:  # binary data
            data = data_info['content']

        # 反序列化metadata
        metadata = json.loads(data_dict.pop('metadata'))

        return cls(
            data=data,
            metadata=metadata,
            data_type=data_info['type'],
            **data_dict
        )


class CacheManager(ICacheManager):
    """缓存管理器实现"""

    def __init__(self, cache_path: Optional[str] = None):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        self.cache_path = Path(cache_path or self.settings.database.cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        # 线程锁
        self._lock = threading.RLock()

        # 初始化数据库
        self._init_database()

    def _init_database(self):
        """初始化数据库"""
        with self._get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP,
                    source_id TEXT,
                    metadata TEXT,
                    data_type TEXT DEFAULT 'dataframe'
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_stats (
                    id INTEGER PRIMARY KEY,
                    total_entries INTEGER DEFAULT 0,
                    total_size_bytes INTEGER DEFAULT 0,
                    hit_count INTEGER DEFAULT 0,
                    miss_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 初始化统计信息
            cursor = conn.execute('SELECT COUNT(*) FROM cache_stats')
            if cursor.fetchone()[0] == 0:
                conn.execute('''
                    INSERT INTO cache_stats (total_entries, total_size_bytes, hit_count, miss_count)
                    VALUES (0, 0, 0, 0)
                ''')

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """获取数据库连接"""
        with self._lock:
            conn = sqlite3.connect(
                str(self.cache_path),
                check_same_thread=False,
                timeout=self.settings.database.query_timeout
            )
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()

    def cache_key(self, source_id: str, query: DataQuery) -> str:
        """生成缓存键"""
        key_data = {
            'source_id': source_id,
            'start_date': query.start_date.isoformat(),
            'end_date': query.end_date.isoformat(),
            'symbols': sorted(query.symbols or []),
            'fields': sorted(query.fields or []),
            'frequency': query.frequency.value if query.frequency else None
        }

        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    @handle_errors(ErrorCategory.CACHE)
    async def get_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """获取缓存数据"""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute('''
                    SELECT * FROM cache_entries
                    WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)
                ''', (cache_key, datetime.now()))

                row = cursor.fetchone()

                if row:
                    # 更新访问统计
                    conn.execute('''
                        UPDATE cache_entries
                        SET access_count = access_count + 1,
                            last_accessed = ?
                        WHERE key = ?
                    ''', (datetime.now(), cache_key))

                    # 更新全局统计
                    conn.execute('''
                        UPDATE cache_stats
                        SET hit_count = hit_count + 1,
                            updated_at = ?
                        WHERE id = 1
                    ''', (datetime.now(),))

                    conn.commit()

                    # 反序列化数据
                    entry = CacheEntry.from_dict(dict(row))
                    self.logger.debug(f"Cache hit for key: {cache_key}")

                    if entry.data_type == 'dataframe':
                        return entry.data
                    else:
                        return entry.data
                else:
                    # 更新miss统计
                    conn.execute('''
                        UPDATE cache_stats
                        SET miss_count = miss_count + 1,
                            updated_at = ?
                        WHERE id = 1
                    ''', (datetime.now(),))

                    conn.commit()
                    self.logger.debug(f"Cache miss for key: {cache_key}")
                    return None

        except Exception as e:
            self.logger.error(f"Error getting cached data: {e}", cache_key=cache_key)
            return None

    @handle_errors(ErrorCategory.CACHE)
    async def set_cached_data(self, cache_key: str, data: pd.DataFrame,
                            expiry_hours: int = 24, source_id: Optional[str] = None) -> bool:
        """设置缓存数据"""
        try:
            expires_at = datetime.now() + timedelta(hours=expiry_hours) if expiry_hours > 0 else None

            # 创建缓存条目
            entry = CacheEntry(
                key=cache_key,
                data=data,
                created_at=datetime.now(),
                expires_at=expires_at,
                source_id=source_id,
                data_type='dataframe' if isinstance(data, pd.DataFrame) else 'dict'
            )

            with self._get_connection() as conn:
                # 序列化数据
                entry_dict = entry.to_dict()

                # 计算数据大小
                data_size = len(json.dumps(entry_dict['data']).encode())

                # 插入或更新缓存条目
                conn.execute('''
                    INSERT OR REPLACE INTO cache_entries
                    (key, data, created_at, expires_at, access_count, last_accessed, source_id, metadata, data_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry.key,
                    json.dumps(entry_dict['data']),
                    entry.created_at,
                    entry.expires_at,
                    entry.access_count,
                    entry.last_accessed,
                    entry.source_id,
                    entry_dict['metadata'],
                    entry.data_type
                ))

                # 更新全局统计
                cursor = conn.execute('SELECT total_entries FROM cache_stats WHERE id = 1')
                total_entries = cursor.fetchone()[0]

                conn.execute('''
                    UPDATE cache_stats
                    SET total_entries = total_entries + 1,
                        total_size_bytes = total_size_bytes + ?,
                        updated_at = ?
                    WHERE id = 1
                ''', (data_size, datetime.now()))

                conn.commit()

            self.logger.debug(f"Cached data for key: {cache_key}", size_bytes=data_size)
            return True

        except Exception as e:
            self.logger.error(f"Error setting cached data: {e}", cache_key=cache_key)
            return False

    @handle_errors(ErrorCategory.CACHE)
    def clear_cache(self, pattern: Optional[str] = None) -> int:
        """清除缓存"""
        try:
            with self._get_connection() as conn:
                if pattern:
                    # 按模式清除
                    cursor = conn.execute('DELETE FROM cache_entries WHERE key LIKE ?', (f"%{pattern}%",))
                else:
                    # 清除所有缓存
                    cursor = conn.execute('DELETE FROM cache_entries')

                deleted_count = cursor.rowcount

                # 重置统计信息
                conn.execute('''
                    UPDATE cache_stats
                    SET total_entries = 0,
                        total_size_bytes = 0,
                        updated_at = ?
                    WHERE id = 1
                ''', (datetime.now(),))

                conn.commit()

                self.logger.info(f"Cleared {deleted_count} cache entries", pattern=pattern)
                return deleted_count

        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}", pattern=pattern)
            return 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        try:
            with self._get_connection() as conn:
                # 基本统计
                cursor = conn.execute('SELECT * FROM cache_stats WHERE id = 1')
                stats_row = cursor.fetchone()

                if stats_row:
                    basic_stats = dict(stats_row)
                else:
                    basic_stats = {
                        'total_entries': 0,
                        'total_size_bytes': 0,
                        'hit_count': 0,
                        'miss_count': 0
                    }

                # 计算命中率
                total_requests = basic_stats['hit_count'] + basic_stats['miss_count']
                hit_rate = basic_stats['hit_count'] / total_requests if total_requests > 0 else 0

                # 按数据源统计
                cursor = conn.execute('''
                    SELECT source_id, COUNT(*) as count, SUM(access_count) as total_accesses
                    FROM cache_entries
                    WHERE source_id IS NOT NULL
                    GROUP BY source_id
                ''')

                source_stats = [dict(row) for row in cursor.fetchall()]

                # 过期条目统计
                cursor = conn.execute('''
                    SELECT COUNT(*) as expired_count
                    FROM cache_entries
                    WHERE expires_at IS NOT NULL AND expires_at < ?
                ''', (datetime.now(),))

                expired_count = cursor.fetchone()[0]

                return {
                    **basic_stats,
                    'hit_rate': hit_rate,
                    'expired_entries': expired_count,
                    'source_breakdown': source_stats,
                    'cache_file_path': str(self.cache_path),
                    'cache_file_size_mb': self.cache_path.stat().st_size / (1024 * 1024) if self.cache_path.exists() else 0
                }

        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {}

    @handle_errors(ErrorCategory.CACHE)
    def cleanup_expired_entries(self) -> int:
        """清理过期条目"""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute('''
                    DELETE FROM cache_entries
                    WHERE expires_at IS NOT NULL AND expires_at < ?
                ''', (datetime.now(),))

                deleted_count = cursor.rowcount

                # 更新统计信息
                cursor = conn.execute('SELECT COUNT(*), SUM(LENGTH(data)) FROM cache_entries')
                total_entries, total_size = cursor.fetchone()

                conn.execute('''
                    UPDATE cache_stats
                    SET total_entries = ?,
                        total_size_bytes = COALESCE(?, 0),
                        updated_at = ?
                    WHERE id = 1
                ''', (total_entries, total_size, datetime.now()))

                conn.commit()

                self.logger.info(f"Cleaned up {deleted_count} expired cache entries")
                return deleted_count

        except Exception as e:
            self.logger.error(f"Error cleaning up expired entries: {e}")
            return 0

    def get_all_keys(self) -> List[str]:
        """获取所有缓存键"""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute('SELECT key FROM cache_entries ORDER BY created_at DESC')
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Error getting cache keys: {e}")
            return []

    def get_cache_info(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """获取特定缓存条目信息"""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute('''
                    SELECT * FROM cache_entries WHERE key = ?
                ''', (cache_key,))

                row = cursor.fetchone()
                if row:
                    entry = CacheEntry.from_dict(dict(row))
                    return {
                        'key': entry.key,
                        'source_id': entry.source_id,
                        'created_at': entry.created_at.isoformat(),
                        'expires_at': entry.expires_at.isoformat() if entry.expires_at else None,
                        'access_count': entry.access_count,
                        'last_accessed': entry.last_accessed.isoformat() if entry.last_accessed else None,
                        'is_expired': entry.is_expired(),
                        'data_type': entry.data_type,
                        'metadata': entry.metadata
                    }
                return None
        except Exception as e:
            self.logger.error(f"Error getting cache info: {e}", cache_key=cache_key)
            return None

    def backup_cache(self, backup_path: Optional[str] = None) -> bool:
        """备份缓存数据库"""
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.cache_path.parent / f"cache_backup_{timestamp}.db"

            with self._get_connection() as conn:
                backup_conn = sqlite3.connect(str(backup_path))
                conn.backup(backup_conn)
                backup_conn.close()

            self.logger.info(f"Cache backed up to: {backup_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error backing up cache: {e}")
            return False

    def restore_cache(self, backup_path: str) -> bool:
        """恢复缓存数据库"""
        try:
            backup_file = Path(backup_path)
            if not backup_file.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_path}")

            # 关闭当前连接
            self._init_database()

            # 恢复数据库
            with sqlite3.connect(str(backup_file)) as backup_conn:
                with self._get_connection() as conn:
                    backup_conn.backup(conn)

            self.logger.info(f"Cache restored from: {backup_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error restoring cache: {e}")
            return False


# 异步缓存管理器
class AsyncCacheManager(CacheManager):
    """异步缓存管理器"""

    @handle_errors(ErrorCategory.CACHE)
    async def get_cached_data_async(self, cache_key: str) -> Optional[pd.DataFrame]:
        """异步获取缓存数据"""
        return await self.get_cached_data(cache_key)

    @handle_errors(ErrorCategory.CACHE)
    async def set_cached_data_async(self, cache_key: str, data: pd.DataFrame,
                                 expiry_hours: int = 24, source_id: Optional[str] = None) -> bool:
        """异步设置缓存数据"""
        return await self.set_cached_data(cache_key, data, expiry_hours, source_id)

    async def batch_get_cache(self, cache_keys: List[str]) -> Dict[str, Optional[pd.DataFrame]]:
        """批量获取缓存"""
        results = {}
        tasks = [self.get_cached_data_async(key) for key in cache_keys]
        cached_data_list = await asyncio.gather(*tasks, return_exceptions=True)

        for key, data in zip(cache_keys, cached_data_list):
            if isinstance(data, Exception):
                self.logger.error(f"Error getting cache for key {key}: {data}")
                results[key] = None
            else:
                results[key] = data

        return results

    async def batch_set_cache(self, cache_data: Dict[str, pd.DataFrame],
                            expiry_hours: int = 24, source_id: Optional[str] = None) -> Dict[str, bool]:
        """批量设置缓存"""
        results = {}
        tasks = [
            self.set_cached_data_async(key, data, expiry_hours, source_id)
            for key, data in cache_data.items()
        ]
        success_list = await asyncio.gather(*tasks, return_exceptions=True)

        for key, success in zip(cache_data.keys(), success_list):
            if isinstance(success, Exception):
                self.logger.error(f"Error setting cache for key {key}: {success}")
                results[key] = False
            else:
                results[key] = success

        return results


# 全局缓存管理器实例
cache_manager = AsyncCacheManager()


def get_cache_manager() -> AsyncCacheManager:
    """获取全局缓存管理器"""
    return cache_manager