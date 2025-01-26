import requests
from PIL import Image
import numpy as np
from io import BytesIO
import os
from scipy.ndimage import gaussian_filter, binary_dilation
import random


def get_baidu_maps_snapshot(lat, lng, api_key, zoom=18, size=(640, 640)):
    """
    使用百度地图 API 获取静态地图快照

    参数:
    - lat (float): 纬度
    - lng (float): 经度
    - api_key (str): 百度地图 API Key
    - zoom (int): 缩放级别
    - size (tuple): 图片尺寸

    返回:
    - PIL.Image.Image: 获取的地图图片
    """
    base_url = "https://api.map.baidu.com/staticimage/v2"

    params = {
        "ak": api_key,
        "center": f"{lng},{lat}",  # 百度API顺序是 经度,纬度
        "zoom": zoom,
        "width": size[0],
        "height": size[1],
        "markers": f"{lng},{lat}"  # 在地图上标记当前位置
    }

    response = requests.get(base_url, params=params)

    # 检查返回的内容类型
    content_type = response.headers.get("Content-Type", "")
    if response.status_code == 200 and "image" in content_type:
        return Image.open(BytesIO(response.content))
    else:
        print("百度地图 API 返回错误：", response.text)
        raise Exception(f"百度地图 API 请求失败: {response.status_code}, {response.text}")


def convert_to_minecraft_palette(image, max_distance=30):
    """
    将地图图像转换为 Minecraft 颜色风格

    参数:
    - image (PIL.Image.Image): 输入地图图片
    - max_distance (float): 最大颜色差异

    返回:
    - PIL.Image.Image: 处理后的图片
    """
    palette = {
        'grass': ((211, 248, 226), (108, 152, 47)),
        'water': ((144, 218, 238), (64, 63, 252)),
        'light_grey': ((233, 234, 239), (205, 127, 55))
    }

    original_colors = np.array([color[0] for color in palette.values()])
    minecraft_colors = np.array([color[1] for color in palette.values()])

    if image.mode != 'RGB':
        image = image.convert('RGB')

    image_array = np.array(image)
    pixels = image_array.reshape(-1, 3)

    distances = np.linalg.norm(pixels[:, None] - original_colors[None, :], axis=2)
    min_distances = np.min(distances, axis=1)
    closest_color_indices = np.argmin(distances, axis=1)

    within_distance = min_distances <= max_distance
    new_pixels = pixels.copy()
    new_pixels[within_distance] = minecraft_colors[closest_color_indices[within_distance]]

    return Image.fromarray(new_pixels.reshape(image_array.shape).astype('uint8'))


def generate_minecraft_map(lat, lng, api_key, output_dir="./"):
    """
    生成 Minecraft 风格的地图

    参数:
    - lat (float): 纬度
    - lng (float): 经度
    - api_key (str): 百度地图 API Key
    - output_dir (str): 保存地图的目录

    返回:
    - PIL.Image.Image: 生成的最终地图
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        original_map = get_baidu_maps_snapshot(lat, lng, api_key)

        original_map_path = os.path.join(output_dir, "original_map.png")
        original_map.save(original_map_path)
        print(f"原始地图已保存: {original_map_path}")

        resized_map = original_map.resize((128, 128))
        resized_map_path = os.path.join(output_dir, "resized_map.png")
        resized_map.save(resized_map_path)

        minecraft_map = convert_to_minecraft_palette(resized_map, max_distance=30)
        minecraft_map_path = os.path.join(output_dir, "minecraft_map.png")
        minecraft_map.save(minecraft_map_path)

        scaled_image = minecraft_map.resize((512, 512), resample=Image.NEAREST)
        scaled_image_path = os.path.join(output_dir, "map.png")
        scaled_image.save(scaled_image_path)

        print(f"最终 Minecraft 风格地图已保存: {scaled_image_path}")

        return scaled_image

    except Exception as e:
        print(f"错误: {e}")
        return None


# 运行示例
if __name__ == "__main__":
    latitude = 39.914888  # 替换为你的纬度
    longitude = 116.403874  # 替换为你的经度
    api_key = "gjz6lCfcMGCOLnyjl6z5lycK5nOwOMaW"  # 替换为你的百度 API Key

    final_image = generate_minecraft_map(latitude, longitude, api_key)
    if final_image:
        final_image.show()  # 显示最终生成的地图
