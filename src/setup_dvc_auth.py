"""Configure DVC remote password from Dagshub cached token."""
from dagshub.auth import get_token

token = get_token()
if not token:
    raise SystemExit("No Dagshub token found. Login via: dagshub login")

config_local = """['remote "origin"']
    password = {token}
    ask_password = false
""".format(token=token)

with open(".dvc/config.local", "w", encoding="utf-8") as f:
    f.write(config_local)

print(f"DVC config.local updated (token length: {len(token)})")
