import base64

with open("D:\\resume-screener-mvp\\data\\resumes\\10281555.pdf", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
print(b64)