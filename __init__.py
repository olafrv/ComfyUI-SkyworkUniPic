from .nodes import LoadSkyworkUniPicConfig, LoadSkyworkUniPicImage, LoadSkyworkUniPicCheckpoint, LoadSkyworkUniPicPrompt, Text2Image, SaveSkyworkUniPicImage, ImageEditing, SaveSkyworkUniPicEditImage, Image2Text

NODE_CLASS_MAPPINGS = {
    "LoadSkyworkUniPicConfig": LoadSkyworkUniPicConfig,
    "LoadSkyworkUniPicImage": LoadSkyworkUniPicImage,
    "LoadSkyworkUniPicCheckpoint": LoadSkyworkUniPicCheckpoint,
    "LoadSkyworkUniPicPrompt": LoadSkyworkUniPicPrompt,
    "Text2Image": Text2Image,
    "SaveSkyworkUniPicImage": SaveSkyworkUniPicImage,
    "ImageEditing": ImageEditing,
    "SaveSkyworkUniPicEditImage": SaveSkyworkUniPicEditImage,
    "Image2Text": Image2Text,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSkyworkUniPicConfig": "Load Skywork UniPic Config",
    "LoadSkyworkUniPicImage": "Load Skywork UniPic Image",
    "LoadSkyworkUniPicCheckpoint": "Load Skywork UniPic Checkpoint",
    "LoadSkyworkUniPicPrompt": "Load Skywork UniPic Prompt",
    "Text2Image": "Text To Image Generation",
    "SaveSkyworkUniPicImage": "Save Skywork UniPic Image",
    "ImageEditing": "Image Editing",
    "SaveSkyworkUniPicEditImage": "Save Skywork UniPic Edit Image",
    "Image2Text": "Image To Text Generation",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
