from fastapi.testclient import TestClient
from fastapi import FastAPI
from app import app  # 导入你的 FastAPI 应用实例

# 创建测试客户端
client = TestClient(app)

def test_calculate_route():
    # 请求体数据
    # request_data = {
    #     "start": {
    #         "lat": 53.123,
    #         "lon": 72.21341
    #     },
    #     "end": {
    #         "lat": 31.2304,
    #         "lon": 121.4737
    #     },
    #     "speed": 600,
    #     "time_step": 15,
    #     "mark_time": "2025-01-06 18:33",
    #     "start_time": "2025-01-06 19:44",
    #     "threshold": 0,
    #     "structure_size": 5
    # }

    request_data = {
        "start": {
            "lat": 43.123,
            "lon": 82.21341
        },
        "end": {
            "lat": 41.2304,
            "lon": 91.4737
        },
        "speed": 600,
        "time_step": 15,
        "mark_time": "2025-01-06 18:33",
        "start_time": "2025-01-06 19:44",
        "threshold": 0,
        "structure_size": 5
    }
    
    # 发送POST请求到/calculate-route
    response = client.post("/api/route", json=request_data)
    
    # 断言响应状态码
    assert response.status_code == 200
    
    # 断言返回的 JSON 格式是否正确，检查返回的内容
    response_data = response.json()
    assert "route" in response_data
    assert "summary" in response_data
    assert "start_point" in response_data["route"]
    assert "end_point" in response_data["route"]
    assert "waypoints" in response_data["route"]
    print(response_data)
    # 可选：你可以断言更具体的返回值，如总时间、总距离等
    # assert response_data["summary"]["total_distance_km"] > 0
    # assert response_data["summary"]["estimated_time_min"] > 0
