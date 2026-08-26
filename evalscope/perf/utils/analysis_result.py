import json
import os
import sqlite3

from evalscope.perf.utils.db_util import decode_data

db_path = 'your db path'
# Primitive/container legacy pickles are decoded by the restricted compatibility
# loader automatically.  This switch is only for unusual trusted legacy DBs
# whose pickle payloads contain custom Python objects/classes.
allow_legacy_pickle = os.getenv('EVALSCOPE_ALLOW_LEGACY_PICKLE') == '1'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 获取列名
cursor.execute('PRAGMA table_info(result)')
columns = [info[1] for info in cursor.fetchall()]
print('列名：', columns)

cursor.execute('SELECT * FROM result WHERE success=1 AND first_chunk_latency > 1')
rows = cursor.fetchall()
print(f'len(rows): {len(rows)}')

for row in rows:
    row_dict = dict(zip(columns, row))
    # request is plain JSON text in current DBs; legacy DBs may contain a
    # base64/pickle payload, so use the compatibility decoder as fallback.
    try:
        row_dict['request'] = json.loads(row_dict['request'])
    except (json.JSONDecodeError, TypeError):
        row_dict['request'] = decode_data(row_dict['request'], allow_legacy_pickle=allow_legacy_pickle)
    row_dict['response_messages'] = decode_data(
        row_dict['response_messages'], allow_legacy_pickle=allow_legacy_pickle
    )
    # print(row_dict)
    print(
        f"request_id: {json.loads(row_dict['response_messages'][0])['id']}, first_chunk_latency: {row_dict['first_chunk_latency']}"  # noqa: E501
    )
    # 如果只想看一个可以break
    # break
