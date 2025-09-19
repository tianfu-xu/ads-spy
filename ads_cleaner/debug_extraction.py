import requests
import sys
from data_cleaner import extract_from_image_url

# Hebrew example
a_url = "https://displayads-formats.googleusercontent.com/ads/preview/content.js?client=ads-integrity-transparency&obfuscatedCustomerId=5413877908&creativeId=766975652297&uiFeatures=12,54&adGroupId=187117781990&assets=%3DH4sIAAAAAAAAAOPy5eLkaDlw9MxMFgEOIPNI96Mpb5kFNIHM9vcHlu5hEmADMhdsefDsE6sAI5A5-WHLtVfsAjxA5pcvR_feYRdgQjCZAbmsLhtPAAAA&sig=ACiVB_yMDo82Nwktw1a0FVw6a_TrdpgDEw&htmlParentId=fletch-render-16504531814295992995&responseCallback=fletchCallback16504531814295992995"
b_url = "https://displayads-formats.googleusercontent.com/ads/preview/content.js?client=ads-integrity-transparency&obfuscatedCustomerId=5413877908&creativeId=707624431814&uiFeatures=12,54&adGroupId=165868298376&assets=%3DH4sIAAAAAAAAAOPy5eLkONL9aMpbZgFNIHPDnwUT9rMIMAGZOyFMRiBzM4TJDGReWbtoFZDJA2S2vz-wdA-TABuQ2XLg6JmZLAIcAA5eBpNPAAAA&sig=ACiVB_wYtSANvbS1bDdSqkG6Cy3UgIFhGw&htmlParentId=fletch-render-8691841256298582172&responseCallback=fletchCallback8691841256298582172"
for label, url in [('hebrew', a_url), ('arabic', b_url)]:
    res = extract_from_image_url(requests.Session(), url)
    sys.stdout.buffer.write(f"=== {label} ===\n".encode('utf-8'))
    for k, v in res.items():
        sys.stdout.buffer.write(f"{k}: {v!r}\n".encode('utf-8'))
