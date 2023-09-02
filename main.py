import os
import json
import openai
import tempfile
import streamlit as st
import os
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dataset_classes = ['metal', 'glass', 'paper', 'trash', 'cardboard', 'plastic']

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    
    return data.to(device, non_blocking=True)

def predict_image(img, model):
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)

    prob, preds  = torch.max(yb, dim=1)

    return dataset_classes[preds[0].item()]

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)

    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  
        loss = F.cross_entropy(out, labels) 

        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                   
        loss = F.cross_entropy(out, labels)   
        acc = accuracy(out, labels)        

        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()     

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))

class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        
        self.network = models.resnet50(pretrained=True)
        
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 6)
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))
    
# def create_prompt(prediction):
#     points = """Прием макулатуры: ТОО “Kagazy Recycling”,  http://www.kagazy.kz/gg.html1. мн.Аксай-3а, ул.Толе би (между Момыш улы и ул.Яссауи).2. мн-11, ул.Шаляпина, уг. ул. Алтынсарина (возле остановки).3. мн-4, дом-22 (возле кинотеатра Сары-арка).4. ул.Толе би, уг. ул. Байзакова (возле магазина «Стрела»).5. ул.Жолдасбекова, уг. ул. Мендикулова.6. ул.Тимирязева 81, уг. ул. Ауэзова (во дворе между домами).7. пр. Рыскулова (между улиц Емцова и Тлендиева, район кафе «Уч-Кудук»),8. ул. Емцова, угол проспекта Райымбека (район магазина «Мурагер»),9. ЖК Алмалы, рядом с Алматы-Арена,ТОО «РЕИЗ» (прием макулатуры), ул. Бекмаханова, 93.ТОО «КАРИНА TRADING» (прием макулатуры), ул. Казыбаева, 264 А, info@karina.kz.ТОО ИП Компания Маолин (Бумажный завод) (прием макулатуры) мкр.Мамыр ул. Садовый бульвар, 1 «З».ТОО «Вторсырье-Маркет» (прием макулатуры, полиэтилен), ул. Казыбаева, 26.ИП Михаил (передвижной пункт приема макулатура, полиэтилена).ТОО «ЭкоПромПереработка» (прием и переработка макулатуры (архив, газеты, журналы, брошюры, типографская обрезь, книги и т.д.), Алматинская область, Илийский район, п. Отеген батыр, ул. Калинина, 17 А.Прием металлов:ИП Юнусов (прием черных и цветных металлов) тел: 294-62-05, 294-29-62, ул. Джумабаева, 13, ориентир – Северное кольцо (поворот на Айнабулак).ТОО «Казвторчермет» (прием черных металлов от физических и юридических лиц), http://kvchm.kz, ул.Рыскулова, 69 (Рыскулова-Козыбаева, не доезжая ул.Аэродромная по нижней части дороги).ИП Михаил (передвижной пункт приема металла).ТОО Rapsh (прием черного лома, демонтаж, резка, погрузка, самовывоз). Казахстан, Алматы ул. малая Суюнбая ниже ул. Бекмаханова 96, 050000.Прием стекла:АО «Стекольная компания САФ» (прием банок, бутылок, стеклобоя из белого стекла, принимают в больших объемах примерно от 10 тонн),  Алматы, мкр. Мамыр-4, д.102/1А (филиал:  Илийский район, пос. Первомайский, Промзона),Специализированный магазин «Амиран» (прием стеклянных бутылок от молока «Амиран»):— пр. Алтынсарина, уг. ул. Куанышбаева (рядом с поликлиникой №6).— ул. Розыбакиева, 125/9, уг. ул. Тимирязева,— мкр-н «Самал — 1» (по улице Мендикулова, ниже улицы Жолдасбекова 200 метров),— ул. Туркебаева, 24, (ниже улицы Болотникова, ориентир магазин «Дидар»).Прием пластиковых отходов:ТОО «Kazakhstan Waste Recycling»ТОО “Kagazy Recycling”  (прием пластиковых бутылок)1. мн «Аксай-3а», ул.Толе би (между Момыш улы и ул.Яссауи).2. мн-11, ул.Шаляпина, уг.ул.Алтынсарина (возле остановки).3. мн-4, дом-22 (возле кинотеатра Сары-Арка).4. ул.Толе би, уг.ул.Байзакова (возле магазина «Стрела»).5. ул.Жолдасбекова, уг.ул.Мендикулова.6. ул.Тимирязева 81, уг.ул.Ауэзова (во дворе между домами).7. ул.Ворошилова, 15 А,"""
#     prompt = f"""Создай список короткого определения и короткого определения вредности: {prediction}. Также добавьте из этого списка возможное место пунктов приема вторсырья в Алматы: [{points}]. Не включайте никаких пояснений, предоставьте только соответствующий ответ JSON в этом формате без отклонений. """
#     json_prompt = '''{"d": "definition", "h": "harmfullness", "rst": ["points"]}'''

#     final = prompt + json_prompt

#     return final

def openai_create(prompt):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{"role": "system", "content": 'Представьте, что вы эксперт в области экологии и различных видов мусора, и у вас есть лучшие навыки, чтобы сказать, что с ними делать. '}, 
                {"role": "user", "content": prompt}], 
        temperature=0.4, 
        max_tokens=2048,
        frequency_penalty=3, 
        stop=None
    )

    return response['choices'][0]['message']['content']


def main():
    st.set_page_config(
        page_title="RecycleAI",
        page_icon="📈",
        # layout="wide"
    )

    transformations = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    model = torch.load('final_model_all.pt', map_location=torch.device('cpu'))
    model.eval()

    st.markdown("<h1 style='text-align: center;'>🤖 Recycle AI</h1>", unsafe_allow_html=True)

    
    with st.sidebar.expander("ℹ️ - About App", expanded=True):
        st.write(
            """
            -   **Recycle AI** - is an application that classifies several types of garbage and gives opportunity to chat with them. 
            -   This sub-application is built around Deep Learning models and OpenAI's ChatGPT.
            """
        )

    API_KEY = st.sidebar.text_input("Your ChatGPT API key", type="password")  

    if API_KEY is None:
        st.error("You didn't provide API KEY")

    openai.api_key = API_KEY

    upload_type = st.sidebar.radio(
        "Choose the uploading type of image:",
        ('File', 'Camera Input')
    )

    if upload_type == 'File':
        uploaded_file = st.file_uploader("Upload image file", type=['png', 'jpeg', 'jpg'])
    
        if uploaded_file != None:
            st.markdown("---")

            c1, c2 = st.columns(2, gap='large')

            image = Image.open(uploaded_file)
            trans_image = transformations(image)

            pred = predict_image(trans_image, model)

            # fprompt = create_prompt(pred)
            points = """Waste paper acceptance: Kagazy Recycling LLP, http://www.kagazy.kz/gg.html1. pl. Aksai-3a, Tole bi st. (between Momysh uly and Yassaui st.).2. mn-11, Chaliapin St., corner. st. Altynsarin (near the stop).3. mn-4, house-22 (near the cinema Saryarka).4. Tole bi street, corner st. Bayzakova (near the store "Strela").5. Zholdasbekov st., corner st. Mendikulova.6. St. Timiryazev 81, corner. st. Auezov (in the courtyard between the houses) .7. Ryskulov avenue (between Yemtsov and Tlendiev streets, Uch-Kuduk cafe area), 8. st. Yemtsov, corner of Raiymbek Avenue (the area of the Murager store), 9. Residential complex Almaly, next to Almaty-Arena, LLP "REIZ" (reception of waste paper), st. Bekmakhanov, 93. LLP "KARINA TRADING" (reception of waste paper), st. Kazybayeva, 264 A, info@karina.kz. LLP IP Company Maolin (Paper Mill) (reception of waste paper) microdistrict Mamyr st. Sadovy Boulevard, 1 "Z". LLP "Recycled Market" (waste paper, polyethylene), st. Kazybaeva, 26. IP Mikhail (mobile collection point for waste paper, polyethylene). EcoPromPererabotka LLP (reception and processing of waste paper (archives, newspapers, magazines, brochures, typographic trimmings, books, etc.), Almaty region, Ili district, Otegen batyr village, Kalinina st., 17 A. Reception of metals: IP Yunusov (reception of ferrous and non-ferrous metals) tel: 294-62-05, 294-29-62, Dzhumabaeva st., 13, landmark - Northern Ring (turn on Ainabulak). LLP "Kazvtorchermet" (acceptance of ferrous metals from individuals and legal entities), http://kvchm.kz, Ryskulov St., 69 (Ryskulova-Kozybaeva, before reaching Aerodromnaya St. along the lower part of the road). IP Mikhail (mobile metal reception point).Rapsh LLP (reception of ferrous scrap, dismantling, cutting, loading, self-delivery) Kazakhstan, Almaty Malaya Suyunbaya street below Bekmakhanov street 96, 050000. , bottles, white glass cullet, accepted in large volumes from about 10 tons), Almaty, microdistrict Mamyr-4, 102/1A (branch: Iliysky district, pos. Pervomaisky, Industrial zone), Specialized store "Amiran" (reception of glass bottles from milk "Amiran"): - Altynsarin Ave., corner. st. Kuanyshbaeva (next to the polyclinic No. 6). - st. Rozybakiev, 125/9, corner. st. Timiryazev, - microdistrict "Samal - 1" (along Mendikulov street, 200 meters below Zholdasbekov street), - st. Turkebaeva, 24, (below Bolotnikova Street, the reference point is the Didar store). Reception of plastic waste: LLP "Kazakhstan Waste Recycling" LLP "Kagazy Recycling" (reception of plastic bottles)1. mn "Aksai-3a", Tole bi st. (between Momysh uly and Yassaui st.) 2. mn-11, Shalyapin St., corner of Altynsarin St. (near the stop).3. mn-4, house-22 (near the cinema Sary-Arka) .4. Tole bi st., corner of Baizakov st. (near the Strela shop).5. Zholdasbekov St., corner of Mendikulov St. 6. 81 Timiryazev St., corner of Auezov St. (in the courtyard between the houses).7. Voroshilov st., 15 A,"""
            d = openai_create(f"Write extremely briefly what is - {pred}")
            h = openai_create(f"Write extremely briefly why this is harmful - {pred}")
            r = openai_create(f"Write where I can recycle this in Almaty city - {pred}, from this list: {points}. Do not include any explanations, only provide a compliant response")

            with c1:
                st.image(image, width=300, caption='Uploaded image file 🖼️')
                            
            with c2:              
                st.success(f"Object on image is - **{pred}**")
                st.info(d)
                st.error(h)

            with st.expander('ℹ️ - Possible recycling points in Almaty'):
                st.write(r)


            c3, c4 = st.columns(2, gap='small')

            with c3:
                question = st.text_input("Ask any question about classified garbage: ")
            with c4:
                sub = st.button('Submit')

            if sub:
                ans = openai_create(f"Answer extremely briefly to this question - {question}")
            
                st.success(ans)

             

    else:
        uploaded_file = st.camera_input("Take a picture of an object")

        if uploaded_file is not None:
            st.markdown("---")

            c1, c2 = st.columns(2, gap='large')

            image = Image.open(uploaded_file)
            trans_image = transformations(image)

            pred = predict_image(trans_image, model)

            # fprompt = create_prompt(pred)
            points = """Waste paper acceptance: Kagazy Recycling LLP, http://www.kagazy.kz/gg.html1. pl. Aksai-3a, Tole bi st. (between Momysh uly and Yassaui st.).2. mn-11, Chaliapin St., corner. st. Altynsarin (near the stop).3. mn-4, house-22 (near the cinema Saryarka).4. Tole bi street, corner st. Bayzakova (near the store "Strela").5. Zholdasbekov st., corner st. Mendikulova.6. St. Timiryazev 81, corner. st. Auezov (in the courtyard between the houses) .7. Ryskulov avenue (between Yemtsov and Tlendiev streets, Uch-Kuduk cafe area), 8. st. Yemtsov, corner of Raiymbek Avenue (the area of the Murager store), 9. Residential complex Almaly, next to Almaty-Arena, LLP "REIZ" (reception of waste paper), st. Bekmakhanov, 93. LLP "KARINA TRADING" (reception of waste paper), st. Kazybayeva, 264 A, info@karina.kz. LLP IP Company Maolin (Paper Mill) (reception of waste paper) microdistrict Mamyr st. Sadovy Boulevard, 1 "Z". LLP "Recycled Market" (waste paper, polyethylene), st. Kazybaeva, 26. IP Mikhail (mobile collection point for waste paper, polyethylene). EcoPromPererabotka LLP (reception and processing of waste paper (archives, newspapers, magazines, brochures, typographic trimmings, books, etc.), Almaty region, Ili district, Otegen batyr village, Kalinina st., 17 A. Reception of metals: IP Yunusov (reception of ferrous and non-ferrous metals) tel: 294-62-05, 294-29-62, Dzhumabaeva st., 13, landmark - Northern Ring (turn on Ainabulak). LLP "Kazvtorchermet" (acceptance of ferrous metals from individuals and legal entities), http://kvchm.kz, Ryskulov St., 69 (Ryskulova-Kozybaeva, before reaching Aerodromnaya St. along the lower part of the road). IP Mikhail (mobile metal reception point).Rapsh LLP (reception of ferrous scrap, dismantling, cutting, loading, self-delivery) Kazakhstan, Almaty Malaya Suyunbaya street below Bekmakhanov street 96, 050000. , bottles, white glass cullet, accepted in large volumes from about 10 tons), Almaty, microdistrict Mamyr-4, 102/1A (branch: Iliysky district, pos. Pervomaisky, Industrial zone), Specialized store "Amiran" (reception of glass bottles from milk "Amiran"): - Altynsarin Ave., corner. st. Kuanyshbaeva (next to the polyclinic No. 6). - st. Rozybakiev, 125/9, corner. st. Timiryazev, - microdistrict "Samal - 1" (along Mendikulov street, 200 meters below Zholdasbekov street), - st. Turkebaeva, 24, (below Bolotnikova Street, the reference point is the Didar store). Reception of plastic waste: LLP "Kazakhstan Waste Recycling" LLP "Kagazy Recycling" (reception of plastic bottles)1. mn "Aksai-3a", Tole bi st. (between Momysh uly and Yassaui st.) 2. mn-11, Shalyapin St., corner of Altynsarin St. (near the stop).3. mn-4, house-22 (near the cinema Sary-Arka) .4. Tole bi st., corner of Baizakov st. (near the Strela shop).5. Zholdasbekov St., corner of Mendikulov St. 6. 81 Timiryazev St., corner of Auezov St. (in the courtyard between the houses).7. Voroshilov st., 15 A,"""
            d = openai_create(f"Write extremely briefly what is - {pred}")
            h = openai_create(f"Write extremely briefly why this is harmful - {pred}")
            r = openai_create(f"Write where I can recycle this in Almaty city - {pred}, from this list: {points}. Do not include any explanations, only provide a compliant response")

            with c1:
                st.image(image, width=300, caption='Uploaded image file 🖼️')
                            
            with c2:              
                st.success(f"Object on image is - **{pred}**")
                st.info(d)
                st.error(h)
                                
            with st.expander('ℹ️ - Possible recycling points in Almaty'):
                st.write(r)

            c3, c4 = st.columns(2, gap='small')

            with c3:
                question = st.text_input("Ask any question about classified garbage: ")
            with c4:
                sub = st.button('Submit')

            if sub:
                ans = openai_create(f"Answer extremely briefly to this question - {question}")
            
                st.success(ans)

if __name__ == '__main__':
    main()