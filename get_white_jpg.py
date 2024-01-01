from PIL import Image

# 创建一个白色的512x512图片
image = Image.new('RGB', (512, 512), color='white')

# 保存为JPG格式
image.save('white_image.jpg')

print('纯白色的512x512图片已保存为white_image.jpg')
