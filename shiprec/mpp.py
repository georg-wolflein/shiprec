from loguru import logger
import openslide

class MPPExtractionError(Exception):
    """Raised when the Microns Per Pixel (MPP) extraction from the slide's metadata fails"""
    pass

def extract_mpp_from_properties(slide: openslide.OpenSlide) -> float:
    try:
        return float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    except KeyError:
        raise MPPExtractionError("MPP could not be loaded from slide")

def extract_mpp_from_metadata(slide: openslide.OpenSlide) -> float:
    import xml.dom.minidom as minidom
    xml_path = slide.properties['tiff.ImageDescription']
    doc = minidom.parseString(xml_path)
    collection = doc.documentElement
    images = collection.getElementsByTagName("Image")
    pixels = images[0].getElementsByTagName("Pixels")
    mpp = float(pixels[0].getAttribute("PhysicalSizeX"))
    if not mpp:
        raise MPPExtractionError("MPP could not be loaded from metadata")
    return mpp

def extract_mpp_from_comments(slide: openslide.OpenSlide) -> float:
    slide_properties = slide.properties.get('openslide.comment')
    pattern = r'<PixelSizeMicrons>(.*?)</PixelSizeMicrons>'
    match = re.search(pattern, slide_properties)
    if not match:
        raise MPPExtractionError("MPP could not be loaded from comments")
    return match.group(1)

SLIDE_MPP_EXTRACTORS = [
    extract_mpp_from_properties, extract_mpp_from_metadata, extract_mpp_from_comments
]

def get_slide_mpp(slide: openslide.OpenSlide) -> float:
    for extractor in SLIDE_MPP_EXTRACTORS:
        try:
            slide_mpp = extractor(slide)
            logger.info(f"MPP successfully extracted using {extractor.__name__}: {slide_mpp:.3f}")
            return slide_mpp
        except MPPExtractionError:
            logger.info(f"MPP could not be extracted using {extractor.__name__}")
    raise MPPExtractionError("MPP could not be extracted from slide")