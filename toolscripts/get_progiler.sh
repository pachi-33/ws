curl -X POST http://127.0.0.1:8818/start_profile -H "Content-Type: application/json"   -d '{
    "num_steps": 5,
    "activities": ["GPU"]
  }'