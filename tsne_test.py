import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 주어진 3차원 리스트 생성
data_3d = np.random.random((3, 3, 64))  # 예시로 랜덤한 값으로 초기화

# 3차원 데이터를 1차원으로 변환 (64채널 유지)
data_1d = data_3d.reshape(-1, 64)

tsne = TSNE(n_components=2, random_state=42)
data_2d = tsne.fit_transform(data_1d)

# 산점도 그리기 및 점의 색상 지정
num_points = data_2d.shape[0]
colors = np.arange(num_points)

# # colors의 종류를 3개로 한정하여 산점도 그리기 및 점의 색상 지정
# num_points = data_2d.shape[0]
# num_colors = 3
# colors = np.repeat(np.arange(num_colors), num_points // num_colors + 1)[:num_points]

plt.scatter(data_2d[:, 0], data_2d[:, 1], c=colors, cmap='viridis')
plt.colorbar()
# plt.scatter(data_2d[:, 0], data_2d[:, 1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('t-SNE Visualization')
plt.show()







