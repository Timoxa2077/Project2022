import torchvision
input_path = "./new_test/"
output_path = ""
output_name = "sub.csv"
device = "none"
return_file_name = False

model = torchvision.models.vgg19
model_wigth = "./wigths/vgg19_MSE.pth"