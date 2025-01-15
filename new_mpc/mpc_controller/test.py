import torch

# 3열 N행의 데이터 배열 (x, y, 속도)
data = torch.tensor([
    [1.0, 2.0, 0.5],  # 예시 데이터: x, y, 속도
    [3.0, 4.0, 0.8],
    [5.0, 6.0, 1.2],
])

# 동차 좌표 생성: x, y, z(0), w(1)
num_points = data.shape[0]
homogeneous_coordinates = torch.hstack((
    data[:, :2],                          # x, y 좌표
    torch.zeros((num_points, 1)),         # z 좌표 (0으로 채움)
    torch.ones((num_points, 1))           # w 좌표 (1로 채움)
))

print("동차 좌표 배열:")
print(homogeneous_coordinates)
