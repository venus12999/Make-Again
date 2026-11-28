import base64
s=open('template.html').read()
w=base64.b64encode(open('/home/claude/wall/Make-Again-照片墙.html','rb').read()).decode()
s=s.replace('__ASSETS__',open('assets.json').read()).replace('__WALL__',w)
open('Make-Again-时光相册.html','w').write(s)
libs='<script>'+open('lib/three.min.js').read()+'</script>\n<script>'+open('crumple3d.js').read()+'</script>\n'
t=s.replace('<script>\nconst A=',libs+'<script>window.__MODE="tools";</script>\n<script>\nconst A=',1).replace('<title>Make Again · 时光相册</title>','<title>Make Again · 时光小屋</title>')
open('Make-Again-时光小屋.html','w').write(t);print(len(s)//1024,'KB')
