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
classIntrinsic = 'fintrinsic'
classLogical = 'flog'
openTagStd = openTag.format(c = classStd)
openTagString = openTag.format(c = classString)
openTagComment = openTag.format(c = classComment)
openTagCuf = openTag.format(c = classCuf)
openTagPp = openTag.format(c = classPp)
openTagIntrinsic = openTag.format(c = classIntrinsic)
openTagLogical = openTag.format(c = classLogical)

matches = re.findall(r'\"(.+?)\"',content)  # match text between two quotes
for m in matches:
      content = content.replace('\"%s\"' % m, '%s\"%s\"%s' % (openTagString, m, closeTag))  # override text to include tags

matches = re.findall(r'\'(.+?)\'',content)  # match text between two quotes
for m in matches:
      content = content.replace('\'%s\'' % m, '%s\'%s\'%s' % (openTagString, m, closeTag))  # override text to include tags

# match comments
matches_comment = re.findall(r'\!.*?(?=\n|$)', content, re.DOTALL)
for m in matches_comment:
    content = content.replace('{}'.format(m), '{}{}{}'.format(openTagComment, m, closeTag))

# match preprocessor comments
matches_pp = re.findall(r'#.*?(?=\n|$)', content, re.DOTALL)
for m in matches_pp:
    content = content.replace('{}'.format(m), '{}{}{}'.format(openTagPp, m, closeTag))

# match launch config chevrons
matches_chevron = re.findall(r'<<<(.+?)>>>', content, re.DOTALL)
for m in matches_chevron:
    content = content.replace('<<<{}>>>'.format(m), '{}&lt;&lt;&lt;{}&gt;&gt;&gt;{}'.format(openTagCuf, m, closeTag))

stdwords = ['stop' 'advance', 'deallocate', 'STAT', 'allocate', 'logical', 'allocatable', 'contains','value', 'real', 'subroutine', 'program', 'use', 'implicit', 'none', 'type', 'integer', 'if', 'then', 'else', 'elseif', 'end', 'endif', 'do', 'enddo', 'module', 'parameter', 'kind', 'endprogram', 'endmodule', 'call']

for w in stdwords:
    content = re.sub(r'\b{}\b'.format(w), '{o}{w}{c}'.format(o=openTagStd, w=w, c=closeTag), content)

cufwords = ['shared', 'dim3', 'cuda_stream_kind', 'PINNED','cudaEvent', 'pinned', 'global','blockIdx', 'blockDim', 'threadIdx', 'device', 'cudaDeviceProp', 'cudaSuccess', '>>>', '<<<', 'cudafor']
for w in cufwords:
    content = re.sub(r'\b{}\b'.format(w), '{o}{w}{c}'.format(o=openTagCuf, w=w, c=closeTag), content)

cuffuncs = ['cudaDeviceDisablePeerAccess', 'cudaMemcpyPeer', 'cudaDeviceEnablePeerAccess', 'cudaDeviceCanAccessPeer', 'cudaMemGetInfo', 'cudaSetDevice', 'int_ptr_kind', 'syncthreads', 'cudaStreamDestroy', 'cudaStreamCreate', 'cudaMemcpyAsync', 'cudaMemcpy', 'cudaMemcpy2D', 'cudaMemcpy3D', 'cudaEventSynchronize', 'cudaEventDestroy', 'cudaEventElapsedTime', 'cudaEventRecord', 'cudaEventCreate', 'attributes','cudaGetDeviceCount', 'cudaGetDeviceProperties', 'cudaGetErrorString', 'cudaGetLastError', 'cudaDeviceSynchronize']
for fu in cuffuncs:
    content = re.sub(r'\b{}\('.format(fu), '{o}{fu}{c}('.format(o=openTagCuf, fu=fu, c=closeTag), content)

intrinsicwords = ['sizeof', 'sqrt', 'sin', 'cos', 'abs', 'maxval', 'trim', 'allocated', 'any']
for w in intrinsicwords:
    content = re.sub(r'\b{}\b'.format(w), '{o}{w}{c}'.format(o=openTagIntrinsic, w=w, c=closeTag), content)

content = re.sub(r'\bwrite\(', '{o}{w}{c}('.format(o=openTagStd, w='write', c=closeTag), content)

logicalwords = ['and', 'or', 'lt', 'gt', 'not', 'eq', 'ge', 'le' ,'false', 'true']
for w in logicalwords:
    content = re.sub(r'\.{}\.'.format(w), '{o}.{w}.{c}'.format(o=openTagLogical, w=w, c=closeTag), content)

print(content)
