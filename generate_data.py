import numpy as np
import pandas as pd
from faker import Faker
import random
import uuid
from datetime import datetime, timedelta

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

N_ENTITIES = 250
SIM_DAYS = 30
SIM_START = datetime(2026, 6, 1)

RESOURCE_POOL = [f"resource_{i}" for i in range(1, 61)]
AUTH_METHODS = ["password", "token", "certificate", "biometric"]
CITIES = [(fake.city(), round(random.uniform(-90, 90), 4), round(random.uniform(-180, 180), 4)) for _ in range(40)]

ENTITY_TYPES = ["user"] * 7 + ["service_account"] * 2 + ["edge_device"] * 1

def make_profile(entity_id, entity_type):
    home = random.choice(CITIES)
    n_resources = random.randint(3, 8)
    return {
        "entity_id": entity_id,
        "entity_type": entity_type,
        "login_hour_mean": random.uniform(7, 19),
        "login_hour_std": random.uniform(0.5, 2.0),
        "home_city": home[0],
        "home_lat": home[1],
        "home_lon": home[2],
        "resources": random.sample(RESOURCE_POOL, n_resources),
        "auth_method": random.choice(AUTH_METHODS),
        "session_duration_mean": random.uniform(5, 90),
        "os_fw": fake.user_agent() if entity_type != "edge_device" else f"firmware_v{random.randint(1,5)}.{random.randint(0,9)}",
        "mac": fake.mac_address(),
        "sessions_per_day": random.uniform(0.3, 4.0),
    }

profiles = {}
for i in range(N_ENTITIES):
    etype = random.choice(ENTITY_TYPES)
    eid = f"{etype}_{i:04d}"
    profiles[eid] = make_profile(eid, etype)

rows = []

def random_time_on_day(day_offset, hour_mean, hour_std):
    hour = np.clip(np.random.normal(hour_mean, hour_std), 0, 23.98)
    dt = SIM_START + timedelta(days=day_offset, hours=hour)
    return dt

def new_row(entity_id, timestamp, source_ip, geo, resource, auth_method,
            session_duration, command_sequence, device_fingerprint, success, label):
    p = profiles[entity_id]
    rows.append({
        "session_id": str(uuid.uuid4()),
        "entity_id": entity_id,
        "entity_type": p["entity_type"],
        "timestamp": timestamp,
        "source_ip": source_ip,
        "geo_location": geo,
        "resource_accessed": resource,
        "auth_method": auth_method,
        "auth_success": success,
        "session_duration_min": round(session_duration, 2),
        "command_sequence": command_sequence,
        "device_fingerprint": device_fingerprint,
        "label": label,
    })

def normal_session(entity_id, day_offset):
    p = profiles[entity_id]
    ts = random_time_on_day(day_offset, p["login_hour_mean"], p["login_hour_std"])
    resource = random.choice(p["resources"])
    duration = max(1, np.random.normal(p["session_duration_mean"], p["session_duration_mean"] * 0.3))
    cmds = random.sample(["read", "write", "list", "download", "query"], k=random.randint(1, 3))
    new_row(entity_id, ts, fake.ipv4(), p["home_city"], resource, p["auth_method"],
            duration, ",".join(cmds), p["os_fw"], True, "normal")

for eid, p in profiles.items():
    n_sessions = int(p["sessions_per_day"] * SIM_DAYS)
    for _ in range(n_sessions):
        day = random.randint(0, SIM_DAYS - 1)
        normal_session(eid, day)

def inject_brute_force(n_incidents):
    for _ in range(n_incidents):
        eid = random.choice(list(profiles.keys()))
        day = random.randint(0, SIM_DAYS - 1)
        ip = fake.ipv4()
        base_ts = random_time_on_day(day, random.uniform(0, 23), 0.1)
        for j in range(random.randint(20, 50)):
            ts = base_ts + timedelta(seconds=j * random.uniform(1, 5))
            new_row(eid, ts, ip, "unknown", random.choice(RESOURCE_POOL),
                    profiles[eid]["auth_method"], 0.1, "auth_attempt",
                    profiles[eid]["os_fw"], False, "brute_force")

def inject_impossible_travel(n_incidents):
    for _ in range(n_incidents):
        eid = random.choice(list(profiles.keys()))
        p = profiles[eid]
        day = random.randint(0, SIM_DAYS - 1)
        ts1 = random_time_on_day(day, p["login_hour_mean"], p["login_hour_std"])
        far_city = random.choice([c for c in CITIES if c[0] != p["home_city"]])
        ts2 = ts1 + timedelta(minutes=random.randint(5, 45))
        new_row(eid, ts1, fake.ipv4(), p["home_city"], random.choice(p["resources"]),
                p["auth_method"], 10, "read", p["os_fw"], True, "impossible_travel")
        new_row(eid, ts2, fake.ipv4(), far_city[0], random.choice(p["resources"]),
                p["auth_method"], 10, "read", p["os_fw"], True, "impossible_travel")

def inject_credential_stuffing(n_incidents):
    for _ in range(n_incidents):
        ips = [fake.ipv4() for _ in range(random.randint(1, 2))]
        targets = random.sample(list(profiles.keys()), k=random.randint(15, 40))
        day = random.randint(0, SIM_DAYS - 1)
        base_ts = random_time_on_day(day, random.uniform(0, 23), 0.1)
        for j, eid in enumerate(targets):
            ts = base_ts + timedelta(seconds=j * random.uniform(0.5, 3))
            new_row(eid, ts, random.choice(ips), "unknown", random.choice(RESOURCE_POOL),
                    profiles[eid]["auth_method"], 0.1, "auth_attempt",
                    profiles[eid]["os_fw"], False, "credential_stuffing")

def inject_lateral_movement(n_incidents):
    for _ in range(n_incidents):
        eid = random.choice(list(profiles.keys()))
        p = profiles[eid]
        day = random.randint(0, SIM_DAYS - 1)
        base_ts = random_time_on_day(day, p["login_hour_mean"], p["login_hour_std"])
        new_resources = random.sample([r for r in RESOURCE_POOL if r not in p["resources"]], k=random.randint(5, 10))
        for j, r in enumerate(new_resources):
            ts = base_ts + timedelta(minutes=j * random.uniform(1, 4))
            new_row(eid, ts, fake.ipv4(), p["home_city"], r, p["auth_method"],
                    random.uniform(1, 5), "list,read,write", p["os_fw"], True, "lateral_movement")

def inject_device_spoofing(n_incidents):
    for _ in range(n_incidents):
        eid = random.choice([e for e, p in profiles.items() if p["entity_type"] == "edge_device"] or list(profiles.keys()))
        p = profiles[eid]
        day = random.randint(0, SIM_DAYS - 1)
        ts = random_time_on_day(day, p["login_hour_mean"], p["login_hour_std"])
        fake_fw = f"firmware_v{random.randint(6,9)}.{random.randint(0,9)}"
        new_row(eid, ts, fake.ipv4(), p["home_city"], random.choice(p["resources"]),
                p["auth_method"], 5, "read", fake_fw, True, "device_spoofing")

def inject_low_and_slow(n_incidents):
    for _ in range(n_incidents):
        eid = random.choice(list(profiles.keys()))
        p = profiles[eid]
        n_days = random.randint(5, 12)
        days = random.sample(range(SIM_DAYS), k=min(n_days, SIM_DAYS))
        for day in days:
            ts = random_time_on_day(day, random.choice([1, 2, 3, 23]), 0.5)
            new_row(eid, ts, fake.ipv4(), p["home_city"], random.choice(RESOURCE_POOL),
                    p["auth_method"], random.uniform(1, 3), "download",
                    p["os_fw"], True, "low_and_slow_exfil")

def inject_insider_drift(n_incidents):
    for _ in range(n_incidents):
        eid = random.choice(list(profiles.keys()))
        p = profiles[eid]
        n_days = random.randint(10, 20)
        days = sorted(random.sample(range(SIM_DAYS), k=min(n_days, SIM_DAYS)))
        pool = [r for r in RESOURCE_POOL if r not in p["resources"]]
        for k, day in enumerate(days):
            ts = random_time_on_day(day, p["login_hour_mean"], p["login_hour_std"])
            n_new = min(1 + k // 5, len(pool))
            r = random.choice(pool[:max(n_new, 1)])
            new_row(eid, ts, fake.ipv4(), p["home_city"], r, p["auth_method"],
                    random.uniform(2, 6), "read,write", p["os_fw"], True, "insider_drift")

n_normal_rows = len(rows)
target_anomaly_rows = int(n_normal_rows * 0.018)

inject_brute_force(n_incidents=max(3, target_anomaly_rows // 140))
inject_impossible_travel(n_incidents=max(8, target_anomaly_rows // 25))
inject_credential_stuffing(n_incidents=max(2, target_anomaly_rows // 120))
inject_lateral_movement(n_incidents=max(4, target_anomaly_rows // 40))
inject_device_spoofing(n_incidents=max(8, target_anomaly_rows // 20))
inject_low_and_slow(n_incidents=max(4, target_anomaly_rows // 45))
inject_insider_drift(n_incidents=max(3, target_anomaly_rows // 80))

df = pd.DataFrame(rows)
df = df.sort_values("timestamp").reset_index(drop=True)

print("Total rows:", len(df))
print(df["label"].value_counts())
print("Anomaly %:", round(100 * (df["label"] != "normal").sum() / len(df), 3))

df.to_csv("/mnt/user-data/outputs/access_logs_full.csv", index=False)

df_inference = df.drop(columns=["label"])
df_inference.to_csv("/mnt/user-data/outputs/access_logs_inference.csv", index=False)

df_labels = df[["session_id", "label"]]
df_labels.to_csv("/mnt/user-data/outputs/access_logs_ground_truth.csv", index=F