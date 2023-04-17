"""
Image Captioning
@zhs 为图像自动添加描述
"""
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

from jindai import Plugin
from jindai.models import MediaItem, Paragraph
from plugins.imageproc import MediaItemStage


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ImageCaptioning(MediaItemStage):
    """Image Captioning
    @zhs 自动为图片添加描述（英文）
    """
    def __init__(self) -> None:
        super().__init__()
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to(device)

    def resolve_image(self, i: MediaItem, paragraph: Paragraph):
        paragraph.content += '\n' + self.caption(i.image)
        paragraph.save()
        
    def caption(self, image):
        prompt = 'a picture of'
        inputs = self.processor(images=image.convert('RGB'), text=prompt, return_tensors="pt").to(device, torch.float16)
        outputs = self.model.generate(**inputs)[0]
        outputs = self.processor.decode(outputs, skip_special_tokens=True)
        self.logger(outputs)
        return outputs[len(prompt):].strip()


class ImageCaptioningPlugin(Plugin):
    """Image captioning plugin
    """    
    
    def __init__(self, pmanager, **conf) -> None:
        super().__init__(pmanager, **conf)
        self.register_pipelines([ImageCaptioning])
        
