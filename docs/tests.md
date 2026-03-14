
Zwei Tabellen zum Testen:
- `hub_state` — key/value store
- `tool_cache` — gecachte Tool-Responses

Test-Fragen für den Showcase mit dem HUb

**hub_state testen:**
```
[db_query]: SELECT * FROM hub_state
[db_query]: SELECT key, value FROM hub_state WHERE key = 'test'
[db_query]: SELECT count(*) FROM hub_state
```

**tool_cache testen:**
```
[db_query]: SELECT * FROM tool_cache
[db_query]: SELECT tool_name, prompt, provider FROM tool_cache
[db_query]: SELECT count(*) FROM tool_cache
[db_query]: SELECT tool_name, count(*) as calls FROM tool_cache GROUP BY tool_name
[db_query]: SELECT * FROM tool_cache ORDER BY created_at DESC LIMIT 5
```

**Security Test (sollte REJECT geben):**
```
[db_query]: SELECT * FROM users
[db_query]: DROP TABLE hub_state
[db_query]: INSERT INTO hub_state VALUES ('hack', 'test', '2026')
[db_query]: from hub_state
```

Die letzten vier sind wichtig den es zeigt dass der Hub sich schützt! 
