s=open('template.html').read().replace('__ASSETS__',open('assets.json').read())
open('Make-Again-信箱.html','w').write(s);print(len(s)//1024,'KB')
