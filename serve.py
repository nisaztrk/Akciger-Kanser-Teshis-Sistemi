import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
from model import ChestCancerModel # Kendi oluşturduğumuz model.py'den çekiyoruz

# 1. Sınıf isimleri (Notebook'taki sırayla tam olarak aynı)
class_names = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']

# 2. Modeli yükleme
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChestCancerModel(num_classes=4)

model.load_state_dict(torch.load("chest_model_final.pth", map_location=device))
model.eval()

# 3. Tahmin Fonksiyonu
def predict_lung_cancer(inp_img):
    if inp_img is None:
        return "Lütfen bir görsel yükleyin."
    
    # Görseli modelin beklediği formata sokma (Notebook'taki transform ile aynı)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(inp_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.nn.functional.softmax(output[0], dim=0)
        confidences = {class_names[i]: float(prediction[i]) for i in range(len(class_names))}
    
    return confidences

# 4. Gradio Arayüz Tasarımı
interface = gr.Interface(
    fn=predict_lung_cancer,
    inputs=gr.Image(type="pil", label="Akciğer BT Taraması Yükle"),
    outputs=gr.Label(num_top_classes=4, label="Teşhis Sonucu"),
    title="LUNG CANCER DETECTION SYSTEM",
    description="Bu sistem ResNet18 derin öğrenme mimarisini kullanarak akciğer BT taramalarından kanser türü tahmini yapar.",
    theme="soft"
)

if __name__ == "__main__":
    interface.launch(share=True) 