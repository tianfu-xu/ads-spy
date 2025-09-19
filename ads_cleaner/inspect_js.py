import requests
import sys

URL = "https://displayads-formats.googleusercontent.com/ads/preview/content.js?client=ads-integrity-transparency&obfuscatedCustomerId=5413877908&creativeId=766892263621&uiFeatures=12,54&adGroupId=183291744196&assets=%3DH4sIAAAAAAAAAOPy5eLkONL9aMpbZgFNILNv24M_r9gFeIDMDbuPzfvFIsCEYDIDmQ9mznk0hVWAEchsOXD0zEwWAQ4gs_39gaV7mATYALsFDBNPAAAA&sig=ACiVB_w7x6DyTOqlIerOvcsHnss19cYvHg&htmlParentId=fletch-render-6696602267102155186&responseCallback=fletchCallback6696602267102155186"
resp = requests.get(URL, timeout=20)
resp.raise_for_status()
text = resp.text
idx = text.find("final_url")
sys.stdout.buffer.write(text[idx-100:idx+200].encode('utf-8', errors='ignore'))
