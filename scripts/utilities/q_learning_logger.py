"""
Q-Learning Logger - Persist Q-learning steps for FL clients (Docker and local).
Similar to packet_logger: shared_data in Docker, scripts/utilities locally.

Schema keeps only columns that are populated by the unified clients (no empty
convergence / T_calc / legacy state columns).
"""

import sqlite3
from datetime import datetime
import os
from typing import Optional


def get_db_path():
    """Resolve DB path: Docker uses /shared_data, local uses project shared_data or scripts/utilities."""
    if os.path.exists('/app'):
        shared_dir = '/shared_data'
        if not os.path.exists(shared_dir):
            try:
                os.makedirs(shared_dir, exist_ok=True)
            except Exception:
                pass
        client_id = os.getenv('CLIENT_ID', '')
        node_type = os.getenv('NODE_TYPE', 'unknown')
        if node_type == 'client' and client_id:
            candidate = os.path.join(shared_dir, f'q_learning_client_{client_id}.db')
        else:
            candidate = os.path.join(shared_dir, 'q_learning.db')
    else:
        # Local: use project shared_data so GUI can read same path
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        shared_dir = os.path.join(base, 'shared_data')
        client_id = os.getenv('CLIENT_ID', '')
        if client_id:
            candidate = os.path.join(shared_dir, f'q_learning_client_{client_id}.db')
        else:
            candidate = os.path.join(shared_dir, 'q_learning.db')
    if os.path.isdir(candidate):
        candidate = os.path.join(os.path.dirname(__file__), 'q_learning.db')
    if not os.path.exists(os.path.dirname(candidate)):
        try:
            os.makedirs(os.path.dirname(candidate), exist_ok=True)
        except Exception:
            candidate = os.path.abspath(os.path.join(os.path.dirname(__file__), 'q_learning.db'))
    return candidate


DB_PATH = get_db_path()


def _create_slim_table(c: sqlite3.Cursor) -> None:
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS q_learning_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            link_direction TEXT DEFAULT 'uplink',
            round_num INTEGER NOT NULL,
            episode INTEGER NOT NULL,
            state_comm_level TEXT,
            state_resource TEXT,
            state_battery_level TEXT,
            action TEXT NOT NULL,
            reward REAL NOT NULL,
            q_delta REAL,
            epsilon REAL NOT NULL,
            avg_reward_last_100 REAL,
            converged INTEGER DEFAULT 0,
            metric_communication_time REAL,
            metric_success INTEGER,
            metric_cpu_usage REAL,
            metric_memory_usage REAL,
            metric_bandwidth_usage REAL,
            metric_battery_level REAL,
            metric_energy_usage REAL,
            reward_base REAL,
            reward_communication_time REAL,
            reward_resource_penalty REAL,
            reward_battery_penalty REAL,
            reward_total REAL
        )
        """
    )


def _table_columns(conn: sqlite3.Connection, table: str) -> set:
    c = conn.cursor()
    c.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in c.fetchall()}


def _migrate_legacy_to_slim(conn: sqlite3.Connection) -> None:
    """Replace legacy wide table (T_calc, convergence, state_network, …) with slim schema."""
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='q_learning_log'")
    if not c.fetchone():
        _create_slim_table(c)
        return

    cols = _table_columns(conn, "q_learning_log")
    # Already on slim schema
    if "metric_t_calc" not in cols and "state_network" not in cols and "state_comm_level" in cols:
        return
    # Legacy: drop partial new table if a previous migration failed mid-flight
    c.execute("DROP TABLE IF EXISTS q_learning_log_new")
    # Legacy: has fat columns or missing state_comm_level — rebuild from old data if possible
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS q_learning_log_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            link_direction TEXT DEFAULT 'uplink',
            round_num INTEGER NOT NULL,
            episode INTEGER NOT NULL,
            state_comm_level TEXT,
            state_resource TEXT,
            state_battery_level TEXT,
            action TEXT NOT NULL,
            reward REAL NOT NULL,
            q_delta REAL,
            epsilon REAL NOT NULL,
            avg_reward_last_100 REAL,
            converged INTEGER DEFAULT 0,
            metric_communication_time REAL,
            metric_success INTEGER,
            metric_cpu_usage REAL,
            metric_memory_usage REAL,
            metric_bandwidth_usage REAL,
            metric_battery_level REAL,
            metric_energy_usage REAL,
            reward_base REAL,
            reward_communication_time REAL,
            reward_resource_penalty REAL,
            reward_battery_penalty REAL,
            reward_total REAL
        )
        """
    )
    # Map legacy columns into slim row
    has_comm_level = "state_comm_level" in cols
    has_network = "state_network" in cols
    if has_comm_level and has_network:
        sel_comm = (
            "COALESCE(NULLIF(TRIM(state_comm_level), ''), NULLIF(TRIM(state_network), ''), '')"
        )
    elif has_comm_level:
        sel_comm = "COALESCE(NULLIF(TRIM(state_comm_level), ''), '')"
    elif has_network:
        sel_comm = "COALESCE(NULLIF(TRIM(state_network), ''), '')"
    else:
        sel_comm = "''"
    c.execute(f"""
        INSERT INTO q_learning_log_new (
            client_id, timestamp, link_direction, round_num, episode,
            state_comm_level, state_resource, state_battery_level,
            action, reward, q_delta, epsilon, avg_reward_last_100, converged,
            metric_communication_time, metric_success,
            metric_cpu_usage, metric_memory_usage, metric_bandwidth_usage,
            metric_battery_level, metric_energy_usage,
            reward_base, reward_communication_time, reward_resource_penalty, reward_battery_penalty, reward_total
        )
        SELECT
            client_id, timestamp,
            COALESCE(link_direction, 'uplink'),
            round_num, episode,
            {sel_comm},
            COALESCE(state_resource, ''),
            COALESCE(state_battery_level, ''),
            action, reward, q_delta, epsilon, avg_reward_last_100, converged,
            metric_communication_time, metric_success,
            metric_cpu_usage, metric_memory_usage, metric_bandwidth_usage,
            metric_battery_level, metric_energy_usage,
            reward_base, reward_communication_time, reward_resource_penalty, reward_battery_penalty, reward_total
        FROM q_learning_log
    """)
    c.execute("DROP TABLE q_learning_log")
    c.execute("ALTER TABLE q_learning_log_new RENAME TO q_learning_log")


def init_db(db_path: Optional[str] = None):
    """Create or migrate q_learning_log to the slim schema.

    When ``db_path`` is omitted, uses the module default (``DB_PATH`` / env-based).
    Pass an explicit path when opening a specific client's DB (e.g. GUI).
    """
    path = os.path.abspath(db_path) if db_path else DB_PATH
    conn = sqlite3.connect(path)
    try:
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='q_learning_log'")
        if not c.fetchone():
            _create_slim_table(c)
        else:
            cols = _table_columns(conn, "q_learning_log")
            if "metric_t_calc" in cols or "state_network" in cols or "state_comm_level" not in cols:
                _migrate_legacy_to_slim(conn)
        conn.commit()
    finally:
        conn.close()


def log_q_step(
    client_id: int,
    round_num: int,
    episode: int,
    state_comm_level: str,
    state_resource: str,
    state_battery_level: str,
    action: str,
    reward: float,
    q_delta: float,
    epsilon: float,
    avg_reward_last_100: float = None,
    converged: bool = False,
    metric_communication_time: float = None,
    metric_success: bool = None,
    metric_cpu_usage: float = None,
    metric_memory_usage: float = None,
    metric_bandwidth_usage: float = None,
    metric_battery_level: float = None,
    metric_energy_usage: float = None,
    reward_base: float = None,
    reward_communication_time: float = None,
    reward_resource_penalty: float = None,
    reward_battery_penalty: float = None,
    reward_total: float = None,
    link_direction: str = "uplink",
):
    """Log one Q-learning step."""
    effective_reward_total = reward_total if reward_total is not None else reward
    try:
        init_db()
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO q_learning_log (
                client_id, timestamp, link_direction, round_num, episode,
                state_comm_level, state_resource, state_battery_level,
                action, reward, q_delta, epsilon, avg_reward_last_100, converged,
                metric_communication_time, metric_success,
                metric_cpu_usage, metric_memory_usage, metric_bandwidth_usage,
                metric_battery_level, metric_energy_usage,
                reward_base, reward_communication_time, reward_resource_penalty, reward_battery_penalty, reward_total
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                client_id,
                datetime.now().isoformat(),
                (link_direction or "uplink").strip().lower(),
                round_num,
                episode,
                state_comm_level or "",
                state_resource or "",
                state_battery_level or "",
                action,
                reward,
                q_delta if q_delta is not None else 0.0,
                epsilon,
                avg_reward_last_100 if avg_reward_last_100 is not None else 0.0,
                1 if converged else 0,
                metric_communication_time,
                None if metric_success is None else (1 if metric_success else 0),
                metric_cpu_usage,
                metric_memory_usage,
                metric_bandwidth_usage,
                metric_battery_level,
                metric_energy_usage,
                reward_base,
                reward_communication_time,
                reward_resource_penalty,
                reward_battery_penalty,
                effective_reward_total,
            ),
        )
        conn.commit()
        conn.close()
    except sqlite3.OperationalError as e:
        if "no such table" in str(e).lower():
            init_db()
            log_q_step(
                client_id,
                round_num,
                episode,
                state_comm_level,
                state_resource,
                state_battery_level,
                action,
                reward,
                q_delta,
                epsilon,
                avg_reward_last_100=avg_reward_last_100,
                converged=converged,
                metric_communication_time=metric_communication_time,
                metric_success=metric_success,
                metric_cpu_usage=metric_cpu_usage,
                metric_memory_usage=metric_memory_usage,
                metric_bandwidth_usage=metric_bandwidth_usage,
                metric_battery_level=metric_battery_level,
                metric_energy_usage=metric_energy_usage,
                reward_base=reward_base,
                reward_communication_time=reward_communication_time,
                reward_resource_penalty=reward_resource_penalty,
                reward_battery_penalty=reward_battery_penalty,
                reward_total=reward_total,
                link_direction=link_direction,
            )
        else:
            print(f"[QLearningLogger] Error: {e}")
    except Exception as e:
        print(f"[QLearningLogger] Error: {e}")


if __name__ == "__main__":
    init_db()
    print(f"Q-Learning DB: {DB_PATH}")
