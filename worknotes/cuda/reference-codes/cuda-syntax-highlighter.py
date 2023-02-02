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
classStd = 'cstd'
classString = 'cstring'
classComment = 'ccomment'
classCuf = 'cnv'
classPp = 'cpp'
classIntrinsic = 'cintrinsic'
classLogical = 'clog'
classMPI = 'cmpi'
openTagStd = openTag.format(c = classStd)
openTagString = openTag.format(c = classString)
openTagComment = openTag.format(c = classComment)
openTagCuf = openTag.format(c = classCuf)
openTagPp = openTag.format(c = classPp)
openTagIntrinsic = openTag.format(c = classIntrinsic)
openTagLogical = openTag.format(c = classLogical)
openTagMPI = openTag.format(c = classMPI)

content = content.replace("<","&lt;")
content = content.replace(">","&gt;")

matches = re.findall(r'\"(.+?)\"',content)  # match text between two double quotes
for m in matches:
      content = content.replace('\"%s\"' % m, '%s\"%s\"%s' % (openTagString, m, closeTag))  # override text to include tags

matches = re.findall(r'\'(.+?)\'',content)  # match text between two quotes
for m in matches:
      content = content.replace('\'%s\'' % m, '%s\'%s\'%s' % (openTagString, m, closeTag))  # override text to include tags

# match comments
matches_comment = re.findall(r'//.*?(?=\n|$)', content, re.DOTALL)
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

stdwords = ['continue', 'extern', 'const', 'typedef', 'else', 'return', 'break', 'if', 'for', 'int', 'const', 'size_t', 'float', 'double', 'void', 'struct' ,'new', 'free', 'malloc', 'memcpy', 'exit', 'memset']
for w in stdwords:
    content = re.sub(r'\b{}\b'.format(w), '{o}{w}{c}'.format(o=openTagStd, w=w, c=closeTag), content)

cufwords = ['cudaMemcpyHostToDevice', 'cudaMemcpyDeviceToHost', '__shared__', 'dim3', 'cudaStream_t', 'cudaMallocHost','cudaEvent_t', 'cudaMalloc', '__global__','blockIdx', 'blockDim', 'threadIdx', 'device', 'cudaDeviceProp', 'cudaSuccess', '>>>', '<<<']
for w in cufwords:
    content = re.sub(r'\b{}\b'.format(w), '{o}{w}{c}'.format(o=openTagCuf, w=w, c=closeTag), content)

cuffuncs = ['__syncthreads', 'cudaFree', 'cudaMemSet', 'cudaFreeHost', 'cudaGetDevice', 'cudaDeviceDisablePeerAccess', 'cudaMemcpyPeer', 'cudaDeviceEnablePeerAccess', 'cudaDeviceCanAccessPeer', 'cudaMemGetInfo', 'cudaSetDevice', 'int_ptr_kind', 'syncthreads', 'cudaStreamDestroy', 'cudaStreamCreate', 'cudaMemcpyAsync', 'cudaMemcpy', 'cudaMemcpy2D', 'cudaMemcpy3D', 'cudaEventSynchronize', 'cudaEventDestroy', 'cudaEventElapsedTime', 'cudaEventRecord', 'cudaEventCreate', 'cudaGetDeviceCount', 'cudaGetDeviceProperties', 'cudaGetErrorString', 'cudaGetLastError', 'cudaDeviceSynchronize']
for fu in cuffuncs:
    content = re.sub(r'\b{}\('.format(fu), '{o}{fu}{c}('.format(o=openTagCuf, fu=fu, c=closeTag), content)

intrinsicwords = ['sizeof', 'sqrt', 'sin', 'cos', 'abs']
for w in intrinsicwords:
    content = re.sub(r'\b{}\b'.format(w), '{o}{w}{c}'.format(o=openTagIntrinsic, w=w, c=closeTag), content)

content = re.sub(r'\bwrite\(', '{o}{w}{c}('.format(o=openTagStd, w='write', c=closeTag), content)

logicalwords = ['and', '\|\|', 'lt', 'gt', 'not', 'eq', 'ge', 'le' ,'false', 'true']
for w in logicalwords:
    content = re.sub(r'\.{}\.'.format(w), '{o}.{w}.{c}'.format(o=openTagLogical, w=w, c=closeTag), content)

mpiwords = ['MPI_ALLGATHER', 'MPI_COMM_SPLIT', 'MPI_BARRIER', 'MPI_INIT', 'MPI_FINALIZE', 'MPI_COMM_WORLD', 'MPI_COMM_SIZE', 'MPI_COMM_RANK']
for w in mpiwords:
    content = re.sub(r'\b{}\b'.format(w), '{o}{w}{c}'.format(o=openTagMPI, w=w, c=closeTag), content)

print(content)
