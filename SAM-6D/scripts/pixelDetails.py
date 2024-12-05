import cv2
import numpy as np

def main(image_path):
    # 1. 读入图像
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # 打印原图像的通道数和每个像素的位深度
    if image is not None:
        channels = image.shape[2] if len(image.shape) == 3 else 1
        bit_depth = image.dtype.itemsize * 8
        print(f"原图像的通道数: {channels}")
        print(f"每个像素的位深度: {bit_depth} bits")

        # 2. 缩放图像为原来尺寸的0.1
        scaled_image = cv2.resize(image, None, fx=0.1, fy=0.1)

        # 3. 保存缩放后的图像矩阵
        if channels == 1:  # 单通道图像
            save_decimal(scaled_image, 'scaled_image.txt')
        elif channels >= 3:  # 三通道或多通道图像
            for i in range(channels):
                channel_image = scaled_image[:, :, i]
                save_decimal(channel_image, f'scaled_channel_{i}.txt')

        # 4. 显示缩放后的图像
        # 确保图像是16位
        if bit_depth == 16:
            # 提取高8位
            high_bits = (scaled_image >> 8).astype(np.uint8)  # 右移8位
            # 提取低8位
            low_bits = (scaled_image & 0x00FF).astype(np.uint8)  # 与0x00FF按位与
            # 将原始最大值和最小值对应为0-255
            min_val = np.min(image)
            max_val = np.max(image)
            print(f"原始像素最大值: {max_val}, 最小值: {min_val}")
            normalized_image = (image - min_val) / (max_val - min_val) * 255
            normalized_image = normalized_image.astype(np.uint8)
            # 保存归一化后的图像
            cv2.imwrite('high_bits.png', high_bits)
            cv2.imwrite('low_bits.png', low_bits)
            cv2.imwrite('max255_min0.png', normalized_image)
        else:
            cv2.imwrite('8_bits.png', scaled_image)

def save_decimal(image, filename):
    # 将图像矩阵转换为十进制并保存到文件
    with open(filename, 'w') as f:
        for row in image:
            decimal_row = ' '.join(str(pixel) for pixel in row)
            f.write(decimal_row + '\n')

if __name__ == "__main__":
    image_path = '/data/SAM-6D/SAM-6D/Data/Example/depth.png'  # 替换为你的图像路径
    main(image_path)