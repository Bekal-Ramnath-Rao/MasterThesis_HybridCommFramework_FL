"""
Q-Learning Logger - Persist Q-learning steps for FL clients (Docker and local).
Similar to packet_logger: shared_data in Docker, scripts/utilities locally.
"""

import sqlite3
from datetime import datetime
import os


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


def init_db():
    """Create q_learning_log table if not exists."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS q_learning_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            round_num INTEGER NOT NULL,
            episode INTEGER NOT NULL,
            state_network TEXT,
            state_resource TEXT,
            state_model_size TEXT,
            state_mobility TEXT,
            action TEXT NOT NULL,
            reward REAL NOT NULL,
            q_delta REAL,
            epsilon REAL NOT NULL,
            avg_reward_last_100 REAL,
            converged INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()


def log_q_step(
    client_id: int,
    round_num: int,
    episode: int,
    state_network: str,
    state_resource: str,
    state_model_size: str,
    state_mobility: str,
    action: str,
    reward: float,
    q_delta: float,
    epsilon: float,
    avg_reward_last_100: float = None,
    converged: bool = False,
):
    """Log one Q-learning step. Auto-creates table if needed."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            INSERT INTO q_learning_log (
                client_id, timestamp, round_num, episode,
                state_network, state_resource, state_model_size, state_mobility,
                action, reward, q_delta, epsilon, avg_reward_last_100, converged
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            client_id,
            datetime.now().isoformat(),
            round_num,
            episode,
            state_network or '',
            state_resource or '',
            state_model_size or '',
            state_mobility or '',
            action,
            reward,
            q_delta if q_delta is not None else 0.0,
            epsilon,
            avg_reward_last_100 if avg_reward_last_100 is not None else 0.0,
            1 if converged else 0,
        ))
        conn.commit()
        conn.close()
    except sqlite3.OperationalError as e:
        if 'no such table' in str(e):
            init_db()
            log_q_step(
                client_id, round_num, episode,
                state_network, state_resource, state_model_size, state_mobility,
                action, reward, q_delta, epsilon, avg_reward_last_100, converged,
            )
        else:
            print(f"[QLearningLogger] Error: {e}")
    except Exception as e:
        print(f"[QLearningLogger] Error: {e}")


if __name__ == "__main__":
    init_db()
    print(f"Q-Learning DB: {DB_PATH}")
