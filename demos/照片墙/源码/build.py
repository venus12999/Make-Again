import base64
s=open('template.html').read()
mb=base64.b64encode(open('Make-Again-信箱.html','rb').read()).decode()
s=s.replace('__ASSETS__',open('assets.json').read()).replace('__MAILBOX__',mb)
open('Make-Again-照片墙.html','w').write(s)
print(len(s)//1024,'KB')
