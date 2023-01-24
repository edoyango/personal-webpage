import sys
import regex as re

fname = sys.argv[1]
f = open(fname,'r')

content = f.read()

f.close()

newf = []
writeRegex = re.compile(r'write\(*')
openTag = '<span class="{c}">'
closeTag = '</span>'
classStd = 'fstd'
classString = 'fstring'
classComment = 'fcomment'
classCuf = 'fnv'
classPp = 'fpp'
openTagStd = openTag.format(c = classStd)
openTagString = openTag.format(c = classString)
openTagComment = openTag.format(c = classComment)
openTagCuf = openTag.format(c = classCuf)
openTagPp = openTag.format(c = classPp)

matches = re.findall(r'\"(.+?)\"',content)  # match text between two quotes
for m in matches:
      content = content.replace('\"%s\"' % m, '%s\"%s\"%s' % (openTagString, m, closeTag))  # override text to include tags

# match comments
matches_comment = re.findall(r'\!.*?(?=\n|$)', content, re.DOTALL)
for m in matches_comment:
    content = content.replace('{}'.format(m), '{}{}{}'.format(openTagComment, m, closeTag))

# match preprocessor comments
matches_pp = re.findall(r'#.*?(?=\n|$)', content, re.DOTALL)
for m in matches_pp:
    content = content.replace('{}'.format(m), '{}{}{}'.format(openTagPp, m, closeTag))

# match launch config chevrons
matches_chevron = re.findall(r'<<<.*?(?=>)>>>', content, re.DOTALL)
for m in matches_chevron:
    content = content.replace('{}'.format(m), '{}{}{}'.format(openTagCuf, m, closeTag))

stdwords = ['contains','value', 'real', 'subroutine', 'program', 'use', 'implicit', 'none', 'type', 'integer', 'if', 'then', 'else', 'elseif', 'end', 'endif', 'do', 'enddo', 'module', 'parameter', 'kind', 'endprogram', 'endmodule', 'call']

for w in stdwords:
    content = re.sub(r'\b{}\b'.format(w), '{o}{w}{c}'.format(o=openTagStd, w=w, c=closeTag), content)

cufwords = ['global','blockIdx', 'blockDim', 'threadIdx', 'device', 'cudaDeviceProp', 'cudaSuccess', '>>>', '<<<']
for w in cufwords:
    content = re.sub(r'\b{}\b'.format(w), '{o}{w}{c}'.format(o=openTagCuf, w=w, c=closeTag), content)

cuffuncs = ['attributes','cudaGetDeviceCount', 'cudaGetDeviceProperties', 'cudaGetErrorString', 'cudaGetLastError', 'cudaDeviceSynchronize']
for fu in cuffuncs:
    content = re.sub(r'\b{}\('.format(fu), '{o}{fu}{c}('.format(o=openTagCuf, fu=fu, c=closeTag), content)

content = re.sub(r'\bwrite\(', '{o}{w}{c}('.format(o=openTagStd, w='write', c=closeTag), content)

print(content)
