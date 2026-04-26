"""
备份管理器 - 定时备份与恢复

提供自动定时备份和手动备份恢复功能。

Example:
    >>> from su_memory.storage import BackupManager
    >>> manager = BackupManager(db_path="memories.db", interval=3600)
    >>> manager.start()  # 启动定时备份
    >>> backup_path = manager.backup()  # 手动备份
    >>> manager.restore(backup_path)  # 恢复备份
"""

import shutil
import json
import time
import os
import threading
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class BackupInfo:
    """备份信息"""
    path: str
    timestamp: float
    size: int
    db_records: int
    checksum: Optional[str] = None
    
    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp)
    
    @property
    def name(self) -> str:
        return Path(self.path).name
    
    def to_dict(self) -> Dict:
        return {
            "path": self.path,
            "timestamp": self.timestamp,
            "size": self.size,
            "db_records": self.db_records,
            "checksum": self.checksum,
            "datetime": self.datetime.isoformat(),
            "name": self.name,
        }


class BackupManager:
    """
    备份管理器
    
    支持定时自动备份、手动备份、备份恢复和备份列表管理。
    
    Attributes:
        db_path: 数据库文件路径
        backup_dir: 备份存储目录
        interval: 自动备份间隔（秒）
        max_backups: 最大保留备份数
    
    Example:
        >>> manager = BackupManager(interval=3600)  # 1小时自动备份
        >>> manager.start()
        >>> backup_path = manager.backup()
        >>> manager.restore("backup_20240101_120000.db")
        >>> manager.stop()
    """
    
    def __init__(
        self,
        db_path: str = "su_memory.db",
        backup_dir: str = "backups",
        interval: int = 3600,
        max_backups: int = 10,
    ):
        """初始化备份管理器
        
        Args:
            db_path: 数据库文件路径
            backup_dir: 备份存储目录
            interval: 自动备份间隔（秒），默认1小时
            max_backups: 最大保留备份数，超出后自动清理旧备份
        """
        self._db_path = db_path
        self._backup_dir = Path(backup_dir)
        self._interval = interval
        self._max_backups = max_backups
        self._timer: Optional[threading.Timer] = None
        self._running = False
        self._lock = threading.Lock()
        
        # 确保备份目录存在
        self._backup_dir.mkdir(exist_ok=True)
        
        # 加载现有备份信息
        self._backups: List[BackupInfo] = []
        self._load_backup_info()
    
    def _load_backup_info(self):
        """加载备份目录中的备份信息"""
        self._backups = []
        for backup_path in sorted(self._backup_dir.glob("backup_*.db")):
            try:
                stat = backup_path.stat()
                # 尝试读取关联的元数据
                meta_path = backup_path.with_suffix(".meta.json")
                db_records = 0
                if meta_path.exists():
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                        db_records = meta.get("records", 0)
                
                self._backups.append(BackupInfo(
                    path=str(backup_path),
                    timestamp=stat.st_mtime,
                    size=stat.st_size,
                    db_records=db_records,
                ))
            except Exception:
                pass
    
    @property
    def backup_dir(self) -> Path:
        """备份目录路径"""
        return self._backup_dir
    
    @property
    def interval(self) -> int:
        """备份间隔（秒）"""
        return self._interval
    
    @property
    def max_backups(self) -> int:
        """最大备份数"""
        return self._max_backups
    
    def backup(self, name: Optional[str] = None) -> str:
        """执行备份
        
        Args:
            name: 自定义备份文件名（不含扩展名）
        
        Returns:
            备份文件路径
        
        Raises:
            FileNotFoundError: 数据库文件不存在
        """
        if not os.path.exists(self._db_path):
            raise FileNotFoundError(f"Database file not found: {self._db_path}")
        
        with self._lock:
            # 生成备份文件名
            if name:
                backup_name = f"backup_{name}.db"
            else:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                backup_name = f"backup_{timestamp}.db"
            
            backup_path = self._backup_dir / backup_name
            
            # 复制数据库文件
            shutil.copy2(self._db_path, backup_path)
            
            # 统计记录数并保存元数据
            db_records = self._get_record_count()
            meta = {
                "created": time.time(),
                "db_path": self._db_path,
                "records": db_records,
                "original_size": os.path.getsize(self._db_path),
            }
            
            meta_path = backup_path.with_suffix(".meta.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            
            # 更新备份列表
            backup_info = BackupInfo(
                path=str(backup_path),
                timestamp=time.time(),
                size=os.path.getsize(backup_path),
                db_records=db_records,
            )
            self._backups.append(backup_info)
            
            # 清理旧备份
            self._cleanup_old_backups()
            
            return str(backup_path)
    
    def restore(self, backup_path: str) -> bool:
        """恢复备份
        
        Args:
            backup_path: 备份文件路径
        
        Returns:
            是否成功恢复
        """
        backup_file = Path(backup_path)
        if not backup_file.exists():
            return False
        
        with self._lock:
            try:
                # 备份当前数据库
                if os.path.exists(self._db_path):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    current_backup = self._backup_dir / f"auto_backup_{timestamp}.db"
                    shutil.copy2(self._db_path, current_backup)
                
                # 恢复备份
                shutil.copy2(backup_path, self._db_path)
                return True
            except Exception:
                return False
    
    def restore_to(self, backup_path: str, target_path: str) -> bool:
        """恢复到指定路径
        
        Args:
            backup_path: 备份文件路径
            target_path: 目标路径
        
        Returns:
            是否成功恢复
        """
        backup_file = Path(backup_path)
        if not backup_file.exists():
            return False
        
        try:
            shutil.copy2(backup_path, target_path)
            return True
        except Exception:
            return False
    
    def list_backups(self, limit: Optional[int] = None) -> List[BackupInfo]:
        """列出所有备份
        
        Args:
            limit: 返回数量限制
        
        Returns:
            备份信息列表（按时间降序）
        """
        backups = sorted(self._backups, key=lambda x: x.timestamp, reverse=True)
        if limit:
            return backups[:limit]
        return backups
    
    def get_latest_backup(self) -> Optional[BackupInfo]:
        """获取最新备份
        
        Returns:
            最新备份信息或None
        """
        if not self._backups:
            return None
        return max(self._backups, key=lambda x: x.timestamp)
    
    def delete_backup(self, backup_path: str) -> bool:
        """删除指定备份
        
        Args:
            backup_path: 备份文件路径
        
        Returns:
            是否成功删除
        """
        backup_file = Path(backup_path)
        if not backup_file.exists():
            return False
        
        with self._lock:
            try:
                backup_file.unlink()
                # 删除关联的元数据文件
                meta_file = backup_file.with_suffix(".meta.json")
                if meta_file.exists():
                    meta_file.unlink()
                
                # 从列表中移除
                self._backups = [b for b in self._backups if b.path != backup_path]
                return True
            except Exception:
                return False
    
    def _get_record_count(self) -> int:
        """获取数据库记录数"""
        try:
            import sqlite3
            conn = sqlite3.connect(self._db_path)
            cursor = conn.execute("SELECT COUNT(*) FROM memories")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception:
            return 0
    
    def _cleanup_old_backups(self):
        """清理旧备份"""
        if len(self._backups) <= self._max_backups:
            return
        
        # 按时间排序，删除最旧的
        sorted_backups = sorted(self._backups, key=lambda x: x.timestamp)
        to_delete = sorted_backups[:len(self._backups) - self._max_backups]
        
        for backup in to_delete:
            try:
                Path(backup.path).unlink()
                meta_path = Path(backup.path).with_suffix(".meta.json")
                if meta_path.exists():
                    meta_path.unlink()
            except Exception:
                pass
        
        self._backups = sorted_backups[len(to_delete):]
    
    def start(self):
        """启动定时备份"""
        if self._running:
            return
        
        self._running = True
        self._schedule_next_backup()
    
    def stop(self):
        """停止定时备份"""
        self._running = False
        if self._timer:
            self._timer.cancel()
            self._timer = None
    
    def _schedule_next_backup(self):
        """调度下一次备份"""
        if not self._running:
            return
        
        self._timer = threading.Timer(self._interval, self._on_timer)
        self._timer.daemon = True
        self._timer.start()
    
    def _on_timer(self):
        """定时器回调"""
        try:
            self.backup()
        except Exception:
            pass
        
        # 调度下一次
        self._schedule_next_backup()
    
    def is_running(self) -> bool:
        """是否正在运行"""
        return self._running
    
    def get_stats(self) -> Dict:
        """获取备份统计
        
        Returns:
            统计信息字典
        """
        total_size = sum(b.size for b in self._backups)
        return {
            "backup_count": len(self._backups),
            "total_size": total_size,
            "max_backups": self._max_backups,
            "interval": self._interval,
            "running": self._running,
            "backup_dir": str(self._backup_dir),
        }
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False