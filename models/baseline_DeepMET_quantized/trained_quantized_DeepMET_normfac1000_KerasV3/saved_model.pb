��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
8
Pow
x"T
y"T
z"T"
Ttype:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
�
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
2
Round
x"T
y"T"
Ttype:
2
	
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68��
�
embedding0/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameembedding0/embeddings

)embedding0/embeddings/Read/ReadVariableOpReadVariableOpembedding0/embeddings*
_output_shapes

:*
dtype0
�
embedding1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameembedding1/embeddings

)embedding1/embeddings/Read/ReadVariableOpReadVariableOpembedding1/embeddings*
_output_shapes

:*
dtype0
x
q_dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameq_dense/kernel
q
"q_dense/kernel/Read/ReadVariableOpReadVariableOpq_dense/kernel*
_output_shapes

:*
dtype0
p
q_dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameq_dense/bias
i
 q_dense/bias/Read/ReadVariableOpReadVariableOpq_dense/bias*
_output_shapes
:*
dtype0
�
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma
�
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0
�
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta
�
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0
�
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean
�
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0
�
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance
�
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0
|
q_dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$*!
shared_nameq_dense_1/kernel
u
$q_dense_1/kernel/Read/ReadVariableOpReadVariableOpq_dense_1/kernel*
_output_shapes

:$*
dtype0
t
q_dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*
shared_nameq_dense_1/bias
m
"q_dense_1/bias/Read/ReadVariableOpReadVariableOpq_dense_1/bias*
_output_shapes
:$*
dtype0
�
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*,
shared_namebatch_normalization_1/gamma
�
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:$*
dtype0
�
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*+
shared_namebatch_normalization_1/beta
�
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:$*
dtype0
�
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*2
shared_name#!batch_normalization_1/moving_mean
�
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:$*
dtype0
�
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*6
shared_name'%batch_normalization_1/moving_variance
�
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:$*
dtype0
~
met_weight/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$*"
shared_namemet_weight/kernel
w
%met_weight/kernel/Read/ReadVariableOpReadVariableOpmet_weight/kernel*
_output_shapes

:$*
dtype0
v
met_weight/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namemet_weight/bias
o
#met_weight/bias/Read/ReadVariableOpReadVariableOpmet_weight/bias*
_output_shapes
:*
dtype0
�
met_weight_minus_one/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namemet_weight_minus_one/gamma
�
.met_weight_minus_one/gamma/Read/ReadVariableOpReadVariableOpmet_weight_minus_one/gamma*
_output_shapes
:*
dtype0
�
met_weight_minus_one/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namemet_weight_minus_one/beta
�
-met_weight_minus_one/beta/Read/ReadVariableOpReadVariableOpmet_weight_minus_one/beta*
_output_shapes
:*
dtype0
�
 met_weight_minus_one/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" met_weight_minus_one/moving_mean
�
4met_weight_minus_one/moving_mean/Read/ReadVariableOpReadVariableOp met_weight_minus_one/moving_mean*
_output_shapes
:*
dtype0
�
$met_weight_minus_one/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$met_weight_minus_one/moving_variance
�
8met_weight_minus_one/moving_variance/Read/ReadVariableOpReadVariableOp$met_weight_minus_one/moving_variance*
_output_shapes
:*
dtype0
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
�
Adam/embedding0/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameAdam/embedding0/embeddings/m
�
0Adam/embedding0/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding0/embeddings/m*
_output_shapes

:*
dtype0
�
Adam/embedding1/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameAdam/embedding1/embeddings/m
�
0Adam/embedding1/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding1/embeddings/m*
_output_shapes

:*
dtype0
�
Adam/q_dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/q_dense/kernel/m

)Adam/q_dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/q_dense/kernel/m*
_output_shapes

:*
dtype0
~
Adam/q_dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/q_dense/bias/m
w
'Adam/q_dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/q_dense/bias/m*
_output_shapes
:*
dtype0
�
 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/batch_normalization/gamma/m
�
4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes
:*
dtype0
�
Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/batch_normalization/beta/m
�
3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes
:*
dtype0
�
Adam/q_dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$*(
shared_nameAdam/q_dense_1/kernel/m
�
+Adam/q_dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/q_dense_1/kernel/m*
_output_shapes

:$*
dtype0
�
Adam/q_dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*&
shared_nameAdam/q_dense_1/bias/m
{
)Adam/q_dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/q_dense_1/bias/m*
_output_shapes
:$*
dtype0
�
"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*3
shared_name$"Adam/batch_normalization_1/gamma/m
�
6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes
:$*
dtype0
�
!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*2
shared_name#!Adam/batch_normalization_1/beta/m
�
5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes
:$*
dtype0
�
Adam/met_weight/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$*)
shared_nameAdam/met_weight/kernel/m
�
,Adam/met_weight/kernel/m/Read/ReadVariableOpReadVariableOpAdam/met_weight/kernel/m*
_output_shapes

:$*
dtype0
�
Adam/met_weight/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/met_weight/bias/m
}
*Adam/met_weight/bias/m/Read/ReadVariableOpReadVariableOpAdam/met_weight/bias/m*
_output_shapes
:*
dtype0
�
Adam/embedding0/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameAdam/embedding0/embeddings/v
�
0Adam/embedding0/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding0/embeddings/v*
_output_shapes

:*
dtype0
�
Adam/embedding1/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameAdam/embedding1/embeddings/v
�
0Adam/embedding1/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding1/embeddings/v*
_output_shapes

:*
dtype0
�
Adam/q_dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/q_dense/kernel/v

)Adam/q_dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/q_dense/kernel/v*
_output_shapes

:*
dtype0
~
Adam/q_dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/q_dense/bias/v
w
'Adam/q_dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/q_dense/bias/v*
_output_shapes
:*
dtype0
�
 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/batch_normalization/gamma/v
�
4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes
:*
dtype0
�
Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/batch_normalization/beta/v
�
3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes
:*
dtype0
�
Adam/q_dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$*(
shared_nameAdam/q_dense_1/kernel/v
�
+Adam/q_dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/q_dense_1/kernel/v*
_output_shapes

:$*
dtype0
�
Adam/q_dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*&
shared_nameAdam/q_dense_1/bias/v
{
)Adam/q_dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/q_dense_1/bias/v*
_output_shapes
:$*
dtype0
�
"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*3
shared_name$"Adam/batch_normalization_1/gamma/v
�
6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes
:$*
dtype0
�
!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*2
shared_name#!Adam/batch_normalization_1/beta/v
�
5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes
:$*
dtype0
�
Adam/met_weight/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$*)
shared_nameAdam/met_weight/kernel/v
�
,Adam/met_weight/kernel/v/Read/ReadVariableOpReadVariableOpAdam/met_weight/kernel/v*
_output_shapes

:$*
dtype0
�
Adam/met_weight/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/met_weight/bias/v
}
*Adam/met_weight/bias/v/Read/ReadVariableOpReadVariableOpAdam/met_weight/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer-15
layer-16
layer-17
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
* 
�

embeddings
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses*
�
#
embeddings
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses*
* 
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses* 
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses* 
�
6kernel_quantizer
6bias_quantizer
6kernel_quantizer_internal
6bias_quantizer_internal
7
quantizers

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses*
�
@axis
	Agamma
Bbeta
Cmoving_mean
Dmoving_variance
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses*
�
K
activation
K	quantizer
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses* 
�
6kernel_quantizer
6bias_quantizer
6kernel_quantizer_internal
6bias_quantizer_internal
R
quantizers

Skernel
Tbias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses*
�
[axis
	\gamma
]beta
^moving_mean
_moving_variance
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses*
�
K
activation
K	quantizer
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses* 
�
6kernel_quantizer
6bias_quantizer
6kernel_quantizer_internal
6bias_quantizer_internal
l
quantizers

mkernel
nbias
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses*
�
uaxis
	vgamma
wbeta
xmoving_mean
ymoving_variance
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses*
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�beta_1
�beta_2

�decay
�learning_rate
	�iterm�#m�8m�9m�Am�Bm�Sm�Tm�\m�]m�mm�nm�v�#v�8v�9v�Av�Bv�Sv�Tv�\v�]v�mv�nv�*
�
0
#1
82
93
A4
B5
C6
D7
S8
T9
\10
]11
^12
_13
m14
n15
v16
w17
x18
y19*
Z
0
#1
82
93
A4
B5
S6
T7
\8
]9
m10
n11*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

�serving_default* 
ic
VARIABLE_VALUEembedding0/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*
* 
* 
ic
VARIABLE_VALUEembedding1/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

#0*

#0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses* 
* 
* 
* 

60
61* 
^X
VARIABLE_VALUEq_dense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEq_dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

80
91*

80
91*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*
* 
* 
* 
hb
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
A0
B1
C2
D3*

A0
B1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses* 
* 
* 

60
61* 
`Z
VARIABLE_VALUEq_dense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEq_dense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

S0
T1*

S0
T1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses*
* 
* 
* 
jd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
\0
]1
^2
_3*

\0
]1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses* 
* 
* 

60
61* 
a[
VARIABLE_VALUEmet_weight/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEmet_weight/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

m0
n1*

m0
n1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses*
* 
* 
* 
ic
VARIABLE_VALUEmet_weight_minus_one/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEmet_weight_minus_one/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE met_weight_minus_one/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE$met_weight_minus_one/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
v0
w1
x2
y3*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
KE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
<
C0
D1
^2
_3
v4
w5
x6
y7*
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17*

�0
�1
�2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

C0
D1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

^0
_1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
 
v0
w1
x2
y3*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

�total

�count
�	variables
�	keras_api*
M

�total

�count
�
_fn_kwargs
�	variables
�	keras_api*
M

�total

�count
�
_fn_kwargs
�	variables
�	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�	variables*
��
VARIABLE_VALUEAdam/embedding0/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/embedding1/embeddings/mVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUEAdam/q_dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/q_dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE Adam/batch_normalization/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/batch_normalization/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/q_dense_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/q_dense_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!Adam/batch_normalization_1/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEAdam/met_weight/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUEAdam/met_weight/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/embedding0/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/embedding1/embeddings/vVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUEAdam/q_dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/q_dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE Adam/batch_normalization/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/batch_normalization/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUEAdam/q_dense_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/q_dense_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!Adam/batch_normalization_1/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEAdam/met_weight/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUEAdam/met_weight/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
serving_default_input_cat0Placeholder*'
_output_shapes
:���������d*
dtype0*
shape:���������d
}
serving_default_input_cat1Placeholder*'
_output_shapes
:���������d*
dtype0*
shape:���������d
�
serving_default_input_contPlaceholder*+
_output_shapes
:���������d*
dtype0* 
shape:���������d
�
serving_default_input_pxpyPlaceholder*+
_output_shapes
:���������d*
dtype0* 
shape:���������d
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_cat0serving_default_input_cat1serving_default_input_contserving_default_input_pxpyembedding0/embeddingsembedding1/embeddingsq_dense/kernelq_dense/bias#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betaq_dense_1/kernelq_dense_1/bias%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/betamet_weight/kernelmet_weight/bias$met_weight_minus_one/moving_variancemet_weight_minus_one/gamma met_weight_minus_one/moving_meanmet_weight_minus_one/beta*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *.
f)R'
%__inference_signature_wrapper_4872601
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)embedding0/embeddings/Read/ReadVariableOp)embedding1/embeddings/Read/ReadVariableOp"q_dense/kernel/Read/ReadVariableOp q_dense/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp$q_dense_1/kernel/Read/ReadVariableOp"q_dense_1/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp%met_weight/kernel/Read/ReadVariableOp#met_weight/bias/Read/ReadVariableOp.met_weight_minus_one/gamma/Read/ReadVariableOp-met_weight_minus_one/beta/Read/ReadVariableOp4met_weight_minus_one/moving_mean/Read/ReadVariableOp8met_weight_minus_one/moving_variance/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp0Adam/embedding0/embeddings/m/Read/ReadVariableOp0Adam/embedding1/embeddings/m/Read/ReadVariableOp)Adam/q_dense/kernel/m/Read/ReadVariableOp'Adam/q_dense/bias/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp+Adam/q_dense_1/kernel/m/Read/ReadVariableOp)Adam/q_dense_1/bias/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp,Adam/met_weight/kernel/m/Read/ReadVariableOp*Adam/met_weight/bias/m/Read/ReadVariableOp0Adam/embedding0/embeddings/v/Read/ReadVariableOp0Adam/embedding1/embeddings/v/Read/ReadVariableOp)Adam/q_dense/kernel/v/Read/ReadVariableOp'Adam/q_dense/bias/v/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp+Adam/q_dense_1/kernel/v/Read/ReadVariableOp)Adam/q_dense_1/bias/v/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp,Adam/met_weight/kernel/v/Read/ReadVariableOp*Adam/met_weight/bias/v/Read/ReadVariableOpConst*D
Tin=
;29	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__traced_save_4873491
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding0/embeddingsembedding1/embeddingsq_dense/kernelq_dense/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceq_dense_1/kernelq_dense_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancemet_weight/kernelmet_weight/biasmet_weight_minus_one/gammamet_weight_minus_one/beta met_weight_minus_one/moving_mean$met_weight_minus_one/moving_variancebeta_1beta_2decaylearning_rate	Adam/itertotalcounttotal_1count_1total_2count_2Adam/embedding0/embeddings/mAdam/embedding1/embeddings/mAdam/q_dense/kernel/mAdam/q_dense/bias/m Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/q_dense_1/kernel/mAdam/q_dense_1/bias/m"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/mAdam/met_weight/kernel/mAdam/met_weight/bias/mAdam/embedding0/embeddings/vAdam/embedding1/embeddings/vAdam/q_dense/kernel/vAdam/q_dense/bias/v Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/q_dense_1/kernel/vAdam/q_dense_1/bias/v"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/vAdam/met_weight/kernel/vAdam/met_weight/bias/v*C
Tin<
:28*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference__traced_restore_4873666��
�%
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4870595

inputs5
'assignmovingavg_readvariableop_resource:$7
)assignmovingavg_1_readvariableop_resource:$3
%batchnorm_mul_readvariableop_resource:$/
!batchnorm_readvariableop_resource:$
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:$*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:$�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������$s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:$*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:$*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:$*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *��L=�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:$*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:$x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:$�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *��L=�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:$*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:$~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:$�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:$P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:$~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:$*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:$p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������$h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:$v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:$*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:$
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������$o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������$�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������$: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������$
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4870548

inputs/
!batchnorm_readvariableop_resource:$3
%batchnorm_mul_readvariableop_resource:$1
#batchnorm_readvariableop_1_resource:$1
#batchnorm_readvariableop_2_resource:$
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:$*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:$P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:$~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:$*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:$p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������$z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:$*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:$z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:$*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:$
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������$o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������$�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������$: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������$
 
_user_specified_nameinputs
�
�
+__inference_q_dense_1_layer_call_fn_4872898

inputs
unknown:$
	unknown_0:$
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_q_dense_1_layer_call_and_return_conditional_losses_4870982s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������d$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4873029

inputs/
!batchnorm_readvariableop_resource:$3
%batchnorm_mul_readvariableop_resource:$1
#batchnorm_readvariableop_1_resource:$1
#batchnorm_readvariableop_2_resource:$
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:$*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:$P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:$~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:$*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:$p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������$z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:$*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:$z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:$*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:$
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������$o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������$�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������$: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������$
 
_user_specified_nameinputs
�!
e
I__inference_q_activation_layer_call_and_return_conditional_losses_4872889

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B :Q
Cast_2CastCast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   AG
subSub
Cast_2:y:0sub/y:output:0*
T0*
_output_shapes
: F
Pow_2PowConst:output:0sub:z:0*
T0*
_output_shapes
: D
sub_1Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: _
	LessEqual	LessEqualinputs	sub_1:z:0*
T0*+
_output_shapes
:���������dJ
ReluReluinputs*
T0*+
_output_shapes
:���������dE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?{
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*+
_output_shapes
:���������dD
sub_2Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: _
mulMulones_like:output:0	sub_2:z:0*
T0*+
_output_shapes
:���������dv
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*+
_output_shapes
:���������dT
mul_1MulinputsCast:y:0*
T0*+
_output_shapes
:���������d_
truedivRealDiv	mul_1:z:0
Cast_1:y:0*
T0*+
_output_shapes
:���������dM
NegNegtruediv:z:0*
T0*+
_output_shapes
:���������dQ
RoundRoundtruediv:z:0*
T0*+
_output_shapes
:���������dV
addAddV2Neg:y:0	Round:y:0*
T0*+
_output_shapes
:���������d[
StopGradientStopGradientadd:z:0*
T0*+
_output_shapes
:���������dh
add_1AddV2truediv:z:0StopGradient:output:0*
T0*+
_output_shapes
:���������d_
	truediv_1RealDiv	add_1:z:0Cast:y:0*
T0*+
_output_shapes
:���������dP
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
	truediv_2RealDivtruediv_2/x:output:0Cast:y:0*
T0*
_output_shapes
: L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_3Subsub_3/x:output:0truediv_2:z:0*
T0*
_output_shapes
: p
clip_by_value/MinimumMinimumtruediv_1:z:0	sub_3:z:0*
T0*+
_output_shapes
:���������dT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*+
_output_shapes
:���������da
mul_2Mul
Cast_1:y:0clip_by_value:z:0*
T0*+
_output_shapes
:���������dU
Neg_1NegSelectV2:output:0*
T0*+
_output_shapes
:���������dZ
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*+
_output_shapes
:���������dL
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?_
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*+
_output_shapes
:���������d_
StopGradient_1StopGradient	mul_3:z:0*
T0*+
_output_shapes
:���������dp
add_3AddV2SelectV2:output:0StopGradient_1:output:0*
T0*+
_output_shapes
:���������dU
IdentityIdentity	add_3:z:0*
T0*+
_output_shapes
:���������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
)__inference_q_dense_layer_call_fn_4872670

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_q_dense_layer_call_and_return_conditional_losses_4870831s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
'__inference_model_layer_call_fn_4871661
inputs_0
inputs_1
inputs_2
inputs_3
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:$
	unknown_8:$
	unknown_9:$

unknown_10:$

unknown_11:$

unknown_12:$

unknown_13:$

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_4871158o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|:���������d:���������d:���������d:���������d: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:���������d
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������d
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������d
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:���������d
"
_user_specified_name
inputs/3
�D
�	
B__inference_model_layer_call_and_return_conditional_losses_4871158

inputs
inputs_1
inputs_2
inputs_3$
embedding0_4870711:$
embedding1_4870725:!
q_dense_4870832:
q_dense_4870834:)
batch_normalization_4870837:)
batch_normalization_4870839:)
batch_normalization_4870841:)
batch_normalization_4870843:#
q_dense_1_4870983:$
q_dense_1_4870985:$+
batch_normalization_1_4870988:$+
batch_normalization_1_4870990:$+
batch_normalization_1_4870992:$+
batch_normalization_1_4870994:$$
met_weight_4871134:$ 
met_weight_4871136:*
met_weight_minus_one_4871139:*
met_weight_minus_one_4871141:*
met_weight_minus_one_4871143:*
met_weight_minus_one_4871145:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�"embedding0/StatefulPartitionedCall�"embedding1/StatefulPartitionedCall�"met_weight/StatefulPartitionedCall�,met_weight_minus_one/StatefulPartitionedCall�q_dense/StatefulPartitionedCall�!q_dense_1/StatefulPartitionedCall�
"embedding0/StatefulPartitionedCallStatefulPartitionedCallinputs_2embedding0_4870711*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_embedding0_layer_call_and_return_conditional_losses_4870710�
"embedding1/StatefulPartitionedCallStatefulPartitionedCallinputs_3embedding1_4870725*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_embedding1_layer_call_and_return_conditional_losses_4870724�
concatenate/PartitionedCallPartitionedCall+embedding0/StatefulPartitionedCall:output:0+embedding1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_4870735�
concatenate_1/PartitionedCallPartitionedCallinputs$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_4870744�
q_dense/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0q_dense_4870832q_dense_4870834*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_q_dense_layer_call_and_return_conditional_losses_4870831�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall(q_dense/StatefulPartitionedCall:output:0batch_normalization_4870837batch_normalization_4870839batch_normalization_4870841batch_normalization_4870843*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_4870466�
q_activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_q_activation_layer_call_and_return_conditional_losses_4870895�
!q_dense_1/StatefulPartitionedCallStatefulPartitionedCall%q_activation/PartitionedCall:output:0q_dense_1_4870983q_dense_1_4870985*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_q_dense_1_layer_call_and_return_conditional_losses_4870982�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall*q_dense_1/StatefulPartitionedCall:output:0batch_normalization_1_4870988batch_normalization_1_4870990batch_normalization_1_4870992batch_normalization_1_4870994*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d$*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4870548�
q_activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_q_activation_1_layer_call_and_return_conditional_losses_4871046�
"met_weight/StatefulPartitionedCallStatefulPartitionedCall'q_activation_1/PartitionedCall:output:0met_weight_4871134met_weight_4871136*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_met_weight_layer_call_and_return_conditional_losses_4871133�
,met_weight_minus_one/StatefulPartitionedCallStatefulPartitionedCall+met_weight/StatefulPartitionedCall:output:0met_weight_minus_one_4871139met_weight_minus_one_4871141met_weight_minus_one_4871143met_weight_minus_one_4871145*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_met_weight_minus_one_layer_call_and_return_conditional_losses_4870630�
multiply/PartitionedCallPartitionedCall5met_weight_minus_one/StatefulPartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_multiply_layer_call_and_return_conditional_losses_4871154�
output/PartitionedCallPartitionedCall!multiply/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_4870684n
IdentityIdentityoutput/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall#^embedding0/StatefulPartitionedCall#^embedding1/StatefulPartitionedCall#^met_weight/StatefulPartitionedCall-^met_weight_minus_one/StatefulPartitionedCall ^q_dense/StatefulPartitionedCall"^q_dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|:���������d:���������d:���������d:���������d: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2H
"embedding0/StatefulPartitionedCall"embedding0/StatefulPartitionedCall2H
"embedding1/StatefulPartitionedCall"embedding1/StatefulPartitionedCall2H
"met_weight/StatefulPartitionedCall"met_weight/StatefulPartitionedCall2\
,met_weight_minus_one/StatefulPartitionedCall,met_weight_minus_one/StatefulPartitionedCall2B
q_dense/StatefulPartitionedCallq_dense/StatefulPartitionedCall2F
!q_dense_1/StatefulPartitionedCall!q_dense_1/StatefulPartitionedCall:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs:SO
+
_output_shapes
:���������d
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������d
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������d
 
_user_specified_nameinputs
��
�"
#__inference__traced_restore_4873666
file_prefix8
&assignvariableop_embedding0_embeddings::
(assignvariableop_1_embedding1_embeddings:3
!assignvariableop_2_q_dense_kernel:-
assignvariableop_3_q_dense_bias::
,assignvariableop_4_batch_normalization_gamma:9
+assignvariableop_5_batch_normalization_beta:@
2assignvariableop_6_batch_normalization_moving_mean:D
6assignvariableop_7_batch_normalization_moving_variance:5
#assignvariableop_8_q_dense_1_kernel:$/
!assignvariableop_9_q_dense_1_bias:$=
/assignvariableop_10_batch_normalization_1_gamma:$<
.assignvariableop_11_batch_normalization_1_beta:$C
5assignvariableop_12_batch_normalization_1_moving_mean:$G
9assignvariableop_13_batch_normalization_1_moving_variance:$7
%assignvariableop_14_met_weight_kernel:$1
#assignvariableop_15_met_weight_bias:<
.assignvariableop_16_met_weight_minus_one_gamma:;
-assignvariableop_17_met_weight_minus_one_beta:B
4assignvariableop_18_met_weight_minus_one_moving_mean:F
8assignvariableop_19_met_weight_minus_one_moving_variance:$
assignvariableop_20_beta_1: $
assignvariableop_21_beta_2: #
assignvariableop_22_decay: +
!assignvariableop_23_learning_rate: '
assignvariableop_24_adam_iter:	 #
assignvariableop_25_total: #
assignvariableop_26_count: %
assignvariableop_27_total_1: %
assignvariableop_28_count_1: %
assignvariableop_29_total_2: %
assignvariableop_30_count_2: B
0assignvariableop_31_adam_embedding0_embeddings_m:B
0assignvariableop_32_adam_embedding1_embeddings_m:;
)assignvariableop_33_adam_q_dense_kernel_m:5
'assignvariableop_34_adam_q_dense_bias_m:B
4assignvariableop_35_adam_batch_normalization_gamma_m:A
3assignvariableop_36_adam_batch_normalization_beta_m:=
+assignvariableop_37_adam_q_dense_1_kernel_m:$7
)assignvariableop_38_adam_q_dense_1_bias_m:$D
6assignvariableop_39_adam_batch_normalization_1_gamma_m:$C
5assignvariableop_40_adam_batch_normalization_1_beta_m:$>
,assignvariableop_41_adam_met_weight_kernel_m:$8
*assignvariableop_42_adam_met_weight_bias_m:B
0assignvariableop_43_adam_embedding0_embeddings_v:B
0assignvariableop_44_adam_embedding1_embeddings_v:;
)assignvariableop_45_adam_q_dense_kernel_v:5
'assignvariableop_46_adam_q_dense_bias_v:B
4assignvariableop_47_adam_batch_normalization_gamma_v:A
3assignvariableop_48_adam_batch_normalization_beta_v:=
+assignvariableop_49_adam_q_dense_1_kernel_v:$7
)assignvariableop_50_adam_q_dense_1_bias_v:$D
6assignvariableop_51_adam_batch_normalization_1_gamma_v:$C
5assignvariableop_52_adam_batch_normalization_1_beta_v:$>
,assignvariableop_53_adam_met_weight_kernel_v:$8
*assignvariableop_54_adam_met_weight_bias_v:
identity_56��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*�
value�B�8B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*�
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::*F
dtypes<
:28	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp&assignvariableop_embedding0_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp(assignvariableop_1_embedding1_embeddingsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_q_dense_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_q_dense_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp,assignvariableop_4_batch_normalization_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp+assignvariableop_5_batch_normalization_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp2assignvariableop_6_batch_normalization_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp6assignvariableop_7_batch_normalization_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp#assignvariableop_8_q_dense_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp!assignvariableop_9_q_dense_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_1_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp.assignvariableop_11_batch_normalization_1_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp5assignvariableop_12_batch_normalization_1_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp9assignvariableop_13_batch_normalization_1_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp%assignvariableop_14_met_weight_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp#assignvariableop_15_met_weight_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp.assignvariableop_16_met_weight_minus_one_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp-assignvariableop_17_met_weight_minus_one_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp4assignvariableop_18_met_weight_minus_one_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp8assignvariableop_19_met_weight_minus_one_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_beta_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_beta_2Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_decayIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp!assignvariableop_23_learning_rateIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_iterIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpassignvariableop_27_total_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpassignvariableop_28_count_1Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOpassignvariableop_29_total_2Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOpassignvariableop_30_count_2Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp0assignvariableop_31_adam_embedding0_embeddings_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp0assignvariableop_32_adam_embedding1_embeddings_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_q_dense_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_q_dense_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp4assignvariableop_35_adam_batch_normalization_gamma_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp3assignvariableop_36_adam_batch_normalization_beta_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_q_dense_1_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_q_dense_1_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp6assignvariableop_39_adam_batch_normalization_1_gamma_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp5assignvariableop_40_adam_batch_normalization_1_beta_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp,assignvariableop_41_adam_met_weight_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_met_weight_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp0assignvariableop_43_adam_embedding0_embeddings_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp0assignvariableop_44_adam_embedding1_embeddings_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_q_dense_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_q_dense_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp4assignvariableop_47_adam_batch_normalization_gamma_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp3assignvariableop_48_adam_batch_normalization_beta_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_q_dense_1_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_q_dense_1_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp6assignvariableop_51_adam_batch_normalization_1_gamma_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp5assignvariableop_52_adam_batch_normalization_1_beta_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp,assignvariableop_53_adam_met_weight_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp*assignvariableop_54_adam_met_weight_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �

Identity_55Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_56IdentityIdentity_55:output:0^NoOp_1*
T0*
_output_shapes
: �	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_56Identity_56:output:0*�
_input_shapesr
p: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
7__inference_batch_normalization_1_layer_call_fn_4872996

inputs
unknown:$
	unknown_0:$
	unknown_1:$
	unknown_2:$
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������$*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4870548|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������$: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������$
 
_user_specified_nameinputs
�<
�
D__inference_q_dense_layer_call_and_return_conditional_losses_4870831

inputs)
readvariableop_resource:'
readvariableop_3_resource:
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C[
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes

:N
truedivRealDivmul:z:0Cast:y:0*
T0*
_output_shapes

:@
NegNegtruediv:z:0*
T0*
_output_shapes

:D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:I
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:N
StopGradientStopGradientadd:z:0*
T0*
_output_shapes

:[
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Bv
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:R
mul_1MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C^
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*
_output_shapes

:L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*
_output_shapes

:h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0O
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes

:M
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*
_output_shapes

:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*
_output_shapes

:R
StopGradient_1StopGradient	mul_3:z:0*
T0*
_output_shapes

:h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0j
add_3AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*
_output_shapes

:;
ShapeShapeinputs*
T0*
_output_shapes
:Q
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
numX
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"      S
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:���������_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       c
	transpose	Transpose	add_3:z:0transpose/perm:output:0*
T0*
_output_shapes

:`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ����f
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:h
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:���������S
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :dS
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:v
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:���������dI
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C]
mul_4MulReadVariableOp_3:value:0mul_4/y:output:0*
T0*
_output_shapes
:P
	truediv_2RealDiv	mul_4:z:0
Cast_1:y:0*
T0*
_output_shapes
:@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_4AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_4:z:0*
T0*
_output_shapes
:[
add_5AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Bv
clip_by_value_1/MinimumMinimum	add_5:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:R
mul_5Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   CZ
	truediv_3RealDiv	mul_5:z:0truediv_3/y:output:0*
T0*
_output_shapes
:L
mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_6Mulmul_6/x:output:0truediv_3:z:0*
T0*
_output_shapes
:f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0K
Neg_3NegReadVariableOp_4:value:0*
T0*
_output_shapes
:I
add_6AddV2	Neg_3:y:0	mul_6:z:0*
T0*
_output_shapes
:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_7Mulmul_7/x:output:0	add_6:z:0*
T0*
_output_shapes
:N
StopGradient_3StopGradient	mul_7:z:0*
T0*
_output_shapes
:f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0f
add_7AddV2ReadVariableOp_5:value:0StopGradient_3:output:0*
T0*
_output_shapes
:g
BiasAddBiasAddReshape_2:output:0	add_7:z:0*
T0*+
_output_shapes
:���������dc
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:���������d�
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������d: : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_5:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�<
�
F__inference_q_dense_1_layer_call_and_return_conditional_losses_4872983

inputs)
readvariableop_resource:$'
readvariableop_3_resource:$
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:$*
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C[
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes

:$N
truedivRealDivmul:z:0Cast:y:0*
T0*
_output_shapes

:$@
NegNegtruediv:z:0*
T0*
_output_shapes

:$D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:$I
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:$N
StopGradientStopGradientadd:z:0*
T0*
_output_shapes

:$[
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:$\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Bv
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:$T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:$R
mul_1MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:$P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C^
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*
_output_shapes

:$L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*
_output_shapes

:$h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:$*
dtype0O
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes

:$M
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*
_output_shapes

:$L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*
_output_shapes

:$R
StopGradient_1StopGradient	mul_3:z:0*
T0*
_output_shapes

:$h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:$*
dtype0j
add_3AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*
_output_shapes

:$;
ShapeShapeinputs*
T0*
_output_shapes
:Q
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
numX
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"   $   S
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:���������_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       c
	transpose	Transpose	add_3:z:0transpose/perm:output:0*
T0*
_output_shapes

:$`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ����f
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:$h
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:���������$S
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :dS
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :$�
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:v
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:���������d$I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:$*
dtype0L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C]
mul_4MulReadVariableOp_3:value:0mul_4/y:output:0*
T0*
_output_shapes
:$P
	truediv_2RealDiv	mul_4:z:0
Cast_1:y:0*
T0*
_output_shapes
:$@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:$D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:$K
add_4AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:$N
StopGradient_2StopGradient	add_4:z:0*
T0*
_output_shapes
:$[
add_5AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:$^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Bv
clip_by_value_1/MinimumMinimum	add_5:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:$V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:$R
mul_5Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:$P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   CZ
	truediv_3RealDiv	mul_5:z:0truediv_3/y:output:0*
T0*
_output_shapes
:$L
mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_6Mulmul_6/x:output:0truediv_3:z:0*
T0*
_output_shapes
:$f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:$*
dtype0K
Neg_3NegReadVariableOp_4:value:0*
T0*
_output_shapes
:$I
add_6AddV2	Neg_3:y:0	mul_6:z:0*
T0*
_output_shapes
:$L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_7Mulmul_7/x:output:0	add_6:z:0*
T0*
_output_shapes
:$N
StopGradient_3StopGradient	mul_7:z:0*
T0*
_output_shapes
:$f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:$*
dtype0f
add_7AddV2ReadVariableOp_5:value:0StopGradient_3:output:0*
T0*
_output_shapes
:$g
BiasAddBiasAddReshape_2:output:0	add_7:z:0*
T0*+
_output_shapes
:���������d$c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:���������d$�
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������d: : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_5:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_4872601

input_cat0

input_cat1

input_cont

input_pxpy
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:$
	unknown_8:$
	unknown_9:$

unknown_10:$

unknown_11:$

unknown_12:$

unknown_13:$

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
input_cont
input_pxpy
input_cat0
input_cat1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__wrapped_model_4870442o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|:���������d:���������d:���������d:���������d: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:���������d
$
_user_specified_name
input_cat0:SO
'
_output_shapes
:���������d
$
_user_specified_name
input_cat1:WS
+
_output_shapes
:���������d
$
_user_specified_name
input_cont:WS
+
_output_shapes
:���������d
$
_user_specified_name
input_pxpy
�
�
'__inference_model_layer_call_fn_4871485

input_cont

input_pxpy

input_cat0

input_cat1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:$
	unknown_8:$
	unknown_9:$

unknown_10:$

unknown_11:$

unknown_12:$

unknown_13:$

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
input_cont
input_pxpy
input_cat0
input_cat1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_4871394o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|:���������d:���������d:���������d:���������d: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
+
_output_shapes
:���������d
$
_user_specified_name
input_cont:WS
+
_output_shapes
:���������d
$
_user_specified_name
input_pxpy:SO
'
_output_shapes
:���������d
$
_user_specified_name
input_cat0:SO
'
_output_shapes
:���������d
$
_user_specified_name
input_cat1
�<
�
D__inference_q_dense_layer_call_and_return_conditional_losses_4872755

inputs)
readvariableop_resource:'
readvariableop_3_resource:
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C[
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes

:N
truedivRealDivmul:z:0Cast:y:0*
T0*
_output_shapes

:@
NegNegtruediv:z:0*
T0*
_output_shapes

:D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:I
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:N
StopGradientStopGradientadd:z:0*
T0*
_output_shapes

:[
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Bv
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:R
mul_1MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C^
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*
_output_shapes

:L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*
_output_shapes

:h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0O
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes

:M
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*
_output_shapes

:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*
_output_shapes

:R
StopGradient_1StopGradient	mul_3:z:0*
T0*
_output_shapes

:h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0j
add_3AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*
_output_shapes

:;
ShapeShapeinputs*
T0*
_output_shapes
:Q
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
numX
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"      S
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:���������_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       c
	transpose	Transpose	add_3:z:0transpose/perm:output:0*
T0*
_output_shapes

:`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ����f
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:h
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:���������S
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :dS
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:v
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:���������dI
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C]
mul_4MulReadVariableOp_3:value:0mul_4/y:output:0*
T0*
_output_shapes
:P
	truediv_2RealDiv	mul_4:z:0
Cast_1:y:0*
T0*
_output_shapes
:@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_4AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_4:z:0*
T0*
_output_shapes
:[
add_5AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Bv
clip_by_value_1/MinimumMinimum	add_5:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:R
mul_5Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   CZ
	truediv_3RealDiv	mul_5:z:0truediv_3/y:output:0*
T0*
_output_shapes
:L
mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_6Mulmul_6/x:output:0truediv_3:z:0*
T0*
_output_shapes
:f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0K
Neg_3NegReadVariableOp_4:value:0*
T0*
_output_shapes
:I
add_6AddV2	Neg_3:y:0	mul_6:z:0*
T0*
_output_shapes
:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_7Mulmul_7/x:output:0	add_6:z:0*
T0*
_output_shapes
:N
StopGradient_3StopGradient	mul_7:z:0*
T0*
_output_shapes
:f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0f
add_7AddV2ReadVariableOp_5:value:0StopGradient_3:output:0*
T0*
_output_shapes
:g
BiasAddBiasAddReshape_2:output:0	add_7:z:0*
T0*+
_output_shapes
:���������dc
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:���������d�
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������d: : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_5:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�<
�
G__inference_met_weight_layer_call_and_return_conditional_losses_4871133

inputs)
readvariableop_resource:$'
readvariableop_3_resource:
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:$*
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C[
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes

:$N
truedivRealDivmul:z:0Cast:y:0*
T0*
_output_shapes

:$@
NegNegtruediv:z:0*
T0*
_output_shapes

:$D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:$I
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:$N
StopGradientStopGradientadd:z:0*
T0*
_output_shapes

:$[
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:$\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Bv
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:$T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:$R
mul_1MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:$P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C^
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*
_output_shapes

:$L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*
_output_shapes

:$h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:$*
dtype0O
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes

:$M
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*
_output_shapes

:$L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*
_output_shapes

:$R
StopGradient_1StopGradient	mul_3:z:0*
T0*
_output_shapes

:$h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:$*
dtype0j
add_3AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*
_output_shapes

:$;
ShapeShapeinputs*
T0*
_output_shapes
:Q
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
numX
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"$      S
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����$   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:���������$_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       c
	transpose	Transpose	add_3:z:0transpose/perm:output:0*
T0*
_output_shapes

:$`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"$   ����f
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:$h
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:���������S
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :dS
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:v
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:���������dI
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C]
mul_4MulReadVariableOp_3:value:0mul_4/y:output:0*
T0*
_output_shapes
:P
	truediv_2RealDiv	mul_4:z:0
Cast_1:y:0*
T0*
_output_shapes
:@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_4AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_4:z:0*
T0*
_output_shapes
:[
add_5AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Bv
clip_by_value_1/MinimumMinimum	add_5:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:R
mul_5Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   CZ
	truediv_3RealDiv	mul_5:z:0truediv_3/y:output:0*
T0*
_output_shapes
:L
mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_6Mulmul_6/x:output:0truediv_3:z:0*
T0*
_output_shapes
:f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0K
Neg_3NegReadVariableOp_4:value:0*
T0*
_output_shapes
:I
add_6AddV2	Neg_3:y:0	mul_6:z:0*
T0*
_output_shapes
:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_7Mulmul_7/x:output:0	add_6:z:0*
T0*
_output_shapes
:N
StopGradient_3StopGradient	mul_7:z:0*
T0*
_output_shapes
:f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0f
add_7AddV2ReadVariableOp_5:value:0StopGradient_3:output:0*
T0*
_output_shapes
:g
BiasAddBiasAddReshape_2:output:0	add_7:z:0*
T0*+
_output_shapes
:���������dc
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:���������d�
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������d$: : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_5:S O
+
_output_shapes
:���������d$
 
_user_specified_nameinputs
�
�
7__inference_batch_normalization_1_layer_call_fn_4873009

inputs
unknown:$
	unknown_0:$
	unknown_1:$
	unknown_2:$
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4870595|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������$: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������$
 
_user_specified_nameinputs
�!
e
I__inference_q_activation_layer_call_and_return_conditional_losses_4870895

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B :Q
Cast_2CastCast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   AG
subSub
Cast_2:y:0sub/y:output:0*
T0*
_output_shapes
: F
Pow_2PowConst:output:0sub:z:0*
T0*
_output_shapes
: D
sub_1Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: _
	LessEqual	LessEqualinputs	sub_1:z:0*
T0*+
_output_shapes
:���������dJ
ReluReluinputs*
T0*+
_output_shapes
:���������dE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?{
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*+
_output_shapes
:���������dD
sub_2Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: _
mulMulones_like:output:0	sub_2:z:0*
T0*+
_output_shapes
:���������dv
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*+
_output_shapes
:���������dT
mul_1MulinputsCast:y:0*
T0*+
_output_shapes
:���������d_
truedivRealDiv	mul_1:z:0
Cast_1:y:0*
T0*+
_output_shapes
:���������dM
NegNegtruediv:z:0*
T0*+
_output_shapes
:���������dQ
RoundRoundtruediv:z:0*
T0*+
_output_shapes
:���������dV
addAddV2Neg:y:0	Round:y:0*
T0*+
_output_shapes
:���������d[
StopGradientStopGradientadd:z:0*
T0*+
_output_shapes
:���������dh
add_1AddV2truediv:z:0StopGradient:output:0*
T0*+
_output_shapes
:���������d_
	truediv_1RealDiv	add_1:z:0Cast:y:0*
T0*+
_output_shapes
:���������dP
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
	truediv_2RealDivtruediv_2/x:output:0Cast:y:0*
T0*
_output_shapes
: L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_3Subsub_3/x:output:0truediv_2:z:0*
T0*
_output_shapes
: p
clip_by_value/MinimumMinimumtruediv_1:z:0	sub_3:z:0*
T0*+
_output_shapes
:���������dT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*+
_output_shapes
:���������da
mul_2Mul
Cast_1:y:0clip_by_value:z:0*
T0*+
_output_shapes
:���������dU
Neg_1NegSelectV2:output:0*
T0*+
_output_shapes
:���������dZ
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*+
_output_shapes
:���������dL
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?_
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*+
_output_shapes
:���������d_
StopGradient_1StopGradient	mul_3:z:0*
T0*+
_output_shapes
:���������dp
add_3AddV2SelectV2:output:0StopGradient_1:output:0*
T0*+
_output_shapes
:���������dU
IdentityIdentity	add_3:z:0*
T0*+
_output_shapes
:���������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_layer_call_fn_4872781

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_4870513|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
��
�
B__inference_model_layer_call_and_return_conditional_losses_4872551
inputs_0
inputs_1
inputs_2
inputs_35
#embedding0_embedding_lookup_4872123:5
#embedding1_embedding_lookup_4872129:1
q_dense_readvariableop_resource:/
!q_dense_readvariableop_3_resource:I
;batch_normalization_assignmovingavg_readvariableop_resource:K
=batch_normalization_assignmovingavg_1_readvariableop_resource:G
9batch_normalization_batchnorm_mul_readvariableop_resource:C
5batch_normalization_batchnorm_readvariableop_resource:3
!q_dense_1_readvariableop_resource:$1
#q_dense_1_readvariableop_3_resource:$K
=batch_normalization_1_assignmovingavg_readvariableop_resource:$M
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:$I
;batch_normalization_1_batchnorm_mul_readvariableop_resource:$E
7batch_normalization_1_batchnorm_readvariableop_resource:$4
"met_weight_readvariableop_resource:$2
$met_weight_readvariableop_3_resource:D
6met_weight_minus_one_batchnorm_readvariableop_resource:H
:met_weight_minus_one_batchnorm_mul_readvariableop_resource:F
8met_weight_minus_one_batchnorm_readvariableop_1_resource:F
8met_weight_minus_one_batchnorm_readvariableop_2_resource:
identity��#batch_normalization/AssignMovingAvg�2batch_normalization/AssignMovingAvg/ReadVariableOp�%batch_normalization/AssignMovingAvg_1�4batch_normalization/AssignMovingAvg_1/ReadVariableOp�,batch_normalization/batchnorm/ReadVariableOp�0batch_normalization/batchnorm/mul/ReadVariableOp�%batch_normalization_1/AssignMovingAvg�4batch_normalization_1/AssignMovingAvg/ReadVariableOp�'batch_normalization_1/AssignMovingAvg_1�6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_1/batchnorm/ReadVariableOp�2batch_normalization_1/batchnorm/mul/ReadVariableOp�embedding0/embedding_lookup�embedding1/embedding_lookup�met_weight/ReadVariableOp�met_weight/ReadVariableOp_1�met_weight/ReadVariableOp_2�met_weight/ReadVariableOp_3�met_weight/ReadVariableOp_4�met_weight/ReadVariableOp_5�-met_weight_minus_one/batchnorm/ReadVariableOp�/met_weight_minus_one/batchnorm/ReadVariableOp_1�/met_weight_minus_one/batchnorm/ReadVariableOp_2�1met_weight_minus_one/batchnorm/mul/ReadVariableOp�q_dense/ReadVariableOp�q_dense/ReadVariableOp_1�q_dense/ReadVariableOp_2�q_dense/ReadVariableOp_3�q_dense/ReadVariableOp_4�q_dense/ReadVariableOp_5�q_dense_1/ReadVariableOp�q_dense_1/ReadVariableOp_1�q_dense_1/ReadVariableOp_2�q_dense_1/ReadVariableOp_3�q_dense_1/ReadVariableOp_4�q_dense_1/ReadVariableOp_5b
embedding0/CastCastinputs_2*

DstT0*

SrcT0*'
_output_shapes
:���������d�
embedding0/embedding_lookupResourceGather#embedding0_embedding_lookup_4872123embedding0/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding0/embedding_lookup/4872123*+
_output_shapes
:���������d*
dtype0�
$embedding0/embedding_lookup/IdentityIdentity$embedding0/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding0/embedding_lookup/4872123*+
_output_shapes
:���������d�
&embedding0/embedding_lookup/Identity_1Identity-embedding0/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������db
embedding1/CastCastinputs_3*

DstT0*

SrcT0*'
_output_shapes
:���������d�
embedding1/embedding_lookupResourceGather#embedding1_embedding_lookup_4872129embedding1/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding1/embedding_lookup/4872129*+
_output_shapes
:���������d*
dtype0�
$embedding1/embedding_lookup/IdentityIdentity$embedding1/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding1/embedding_lookup/4872129*+
_output_shapes
:���������d�
&embedding1/embedding_lookup/Identity_1Identity-embedding1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������dY
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate/concatConcatV2/embedding0/embedding_lookup/Identity_1:output:0/embedding1/embedding_lookup/Identity_1:output:0 concatenate/concat/axis:output:0*
N*
T0*+
_output_shapes
:���������d[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_1/concatConcatV2inputs_0concatenate/concat:output:0"concatenate_1/concat/axis:output:0*
N*
T0*+
_output_shapes
:���������dO
q_dense/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :O
q_dense/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :c
q_dense/PowPowq_dense/Pow/x:output:0q_dense/Pow/y:output:0*
T0*
_output_shapes
: U
q_dense/CastCastq_dense/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: v
q_dense/ReadVariableOpReadVariableOpq_dense_readvariableop_resource*
_output_shapes

:*
dtype0R
q_dense/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Cs
q_dense/mulMulq_dense/ReadVariableOp:value:0q_dense/mul/y:output:0*
T0*
_output_shapes

:f
q_dense/truedivRealDivq_dense/mul:z:0q_dense/Cast:y:0*
T0*
_output_shapes

:P
q_dense/NegNegq_dense/truediv:z:0*
T0*
_output_shapes

:T
q_dense/RoundRoundq_dense/truediv:z:0*
T0*
_output_shapes

:a
q_dense/addAddV2q_dense/Neg:y:0q_dense/Round:y:0*
T0*
_output_shapes

:^
q_dense/StopGradientStopGradientq_dense/add:z:0*
T0*
_output_shapes

:s
q_dense/add_1AddV2q_dense/truediv:z:0q_dense/StopGradient:output:0*
T0*
_output_shapes

:d
q_dense/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B�
q_dense/clip_by_value/MinimumMinimumq_dense/add_1:z:0(q_dense/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:\
q_dense/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
q_dense/clip_by_valueMaximum!q_dense/clip_by_value/Minimum:z:0 q_dense/clip_by_value/y:output:0*
T0*
_output_shapes

:j
q_dense/mul_1Mulq_dense/Cast:y:0q_dense/clip_by_value:z:0*
T0*
_output_shapes

:X
q_dense/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Cv
q_dense/truediv_1RealDivq_dense/mul_1:z:0q_dense/truediv_1/y:output:0*
T0*
_output_shapes

:T
q_dense/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?n
q_dense/mul_2Mulq_dense/mul_2/x:output:0q_dense/truediv_1:z:0*
T0*
_output_shapes

:x
q_dense/ReadVariableOp_1ReadVariableOpq_dense_readvariableop_resource*
_output_shapes

:*
dtype0_
q_dense/Neg_1Neg q_dense/ReadVariableOp_1:value:0*
T0*
_output_shapes

:e
q_dense/add_2AddV2q_dense/Neg_1:y:0q_dense/mul_2:z:0*
T0*
_output_shapes

:T
q_dense/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?j
q_dense/mul_3Mulq_dense/mul_3/x:output:0q_dense/add_2:z:0*
T0*
_output_shapes

:b
q_dense/StopGradient_1StopGradientq_dense/mul_3:z:0*
T0*
_output_shapes

:x
q_dense/ReadVariableOp_2ReadVariableOpq_dense_readvariableop_resource*
_output_shapes

:*
dtype0�
q_dense/add_3AddV2 q_dense/ReadVariableOp_2:value:0q_dense/StopGradient_1:output:0*
T0*
_output_shapes

:Z
q_dense/ShapeShapeconcatenate_1/concat:output:0*
T0*
_output_shapes
:a
q_dense/unstackUnpackq_dense/Shape:output:0*
T0*
_output_shapes
: : : *	
num`
q_dense/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"      c
q_dense/unstack_1Unpackq_dense/Shape_1:output:0*
T0*
_output_shapes
: : *	
numf
q_dense/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
q_dense/ReshapeReshapeconcatenate_1/concat:output:0q_dense/Reshape/shape:output:0*
T0*'
_output_shapes
:���������g
q_dense/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       {
q_dense/transpose	Transposeq_dense/add_3:z:0q_dense/transpose/perm:output:0*
T0*
_output_shapes

:h
q_dense/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ����~
q_dense/Reshape_1Reshapeq_dense/transpose:y:0 q_dense/Reshape_1/shape:output:0*
T0*
_output_shapes

:�
q_dense/MatMulMatMulq_dense/Reshape:output:0q_dense/Reshape_1:output:0*
T0*'
_output_shapes
:���������[
q_dense/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d[
q_dense/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
q_dense/Reshape_2/shapePackq_dense/unstack:output:0"q_dense/Reshape_2/shape/1:output:0"q_dense/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:�
q_dense/Reshape_2Reshapeq_dense/MatMul:product:0 q_dense/Reshape_2/shape:output:0*
T0*+
_output_shapes
:���������dQ
q_dense/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :Q
q_dense/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :i
q_dense/Pow_1Powq_dense/Pow_1/x:output:0q_dense/Pow_1/y:output:0*
T0*
_output_shapes
: Y
q_dense/Cast_1Castq_dense/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: v
q_dense/ReadVariableOp_3ReadVariableOp!q_dense_readvariableop_3_resource*
_output_shapes
:*
dtype0T
q_dense/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Cu
q_dense/mul_4Mul q_dense/ReadVariableOp_3:value:0q_dense/mul_4/y:output:0*
T0*
_output_shapes
:h
q_dense/truediv_2RealDivq_dense/mul_4:z:0q_dense/Cast_1:y:0*
T0*
_output_shapes
:P
q_dense/Neg_2Negq_dense/truediv_2:z:0*
T0*
_output_shapes
:T
q_dense/Round_1Roundq_dense/truediv_2:z:0*
T0*
_output_shapes
:c
q_dense/add_4AddV2q_dense/Neg_2:y:0q_dense/Round_1:y:0*
T0*
_output_shapes
:^
q_dense/StopGradient_2StopGradientq_dense/add_4:z:0*
T0*
_output_shapes
:s
q_dense/add_5AddV2q_dense/truediv_2:z:0q_dense/StopGradient_2:output:0*
T0*
_output_shapes
:f
!q_dense/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B�
q_dense/clip_by_value_1/MinimumMinimumq_dense/add_5:z:0*q_dense/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:^
q_dense/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
q_dense/clip_by_value_1Maximum#q_dense/clip_by_value_1/Minimum:z:0"q_dense/clip_by_value_1/y:output:0*
T0*
_output_shapes
:j
q_dense/mul_5Mulq_dense/Cast_1:y:0q_dense/clip_by_value_1:z:0*
T0*
_output_shapes
:X
q_dense/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Cr
q_dense/truediv_3RealDivq_dense/mul_5:z:0q_dense/truediv_3/y:output:0*
T0*
_output_shapes
:T
q_dense/mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?j
q_dense/mul_6Mulq_dense/mul_6/x:output:0q_dense/truediv_3:z:0*
T0*
_output_shapes
:v
q_dense/ReadVariableOp_4ReadVariableOp!q_dense_readvariableop_3_resource*
_output_shapes
:*
dtype0[
q_dense/Neg_3Neg q_dense/ReadVariableOp_4:value:0*
T0*
_output_shapes
:a
q_dense/add_6AddV2q_dense/Neg_3:y:0q_dense/mul_6:z:0*
T0*
_output_shapes
:T
q_dense/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?f
q_dense/mul_7Mulq_dense/mul_7/x:output:0q_dense/add_6:z:0*
T0*
_output_shapes
:^
q_dense/StopGradient_3StopGradientq_dense/mul_7:z:0*
T0*
_output_shapes
:v
q_dense/ReadVariableOp_5ReadVariableOp!q_dense_readvariableop_3_resource*
_output_shapes
:*
dtype0~
q_dense/add_7AddV2 q_dense/ReadVariableOp_5:value:0q_dense/StopGradient_3:output:0*
T0*
_output_shapes
:
q_dense/BiasAddBiasAddq_dense/Reshape_2:output:0q_dense/add_7:z:0*
T0*+
_output_shapes
:���������d�
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
 batch_normalization/moments/meanMeanq_dense/BiasAdd:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(�
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*"
_output_shapes
:�
-batch_normalization/moments/SquaredDifferenceSquaredDifferenceq_dense/BiasAdd:output:01batch_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������d�
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(�
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 n
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *��L=�
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes
:�
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0p
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *��L=�
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:x
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:�
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
#batch_normalization/batchnorm/mul_1Mulq_dense/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������d�
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:�
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������dT
q_activation/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :T
q_activation/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :r
q_activation/PowPowq_activation/Pow/x:output:0q_activation/Pow/y:output:0*
T0*
_output_shapes
: _
q_activation/CastCastq_activation/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: V
q_activation/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :V
q_activation/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :x
q_activation/Pow_1Powq_activation/Pow_1/x:output:0q_activation/Pow_1/y:output:0*
T0*
_output_shapes
: c
q_activation/Cast_1Castq_activation/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: W
q_activation/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @W
q_activation/Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B :k
q_activation/Cast_2Castq_activation/Cast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: W
q_activation/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   An
q_activation/subSubq_activation/Cast_2:y:0q_activation/sub/y:output:0*
T0*
_output_shapes
: m
q_activation/Pow_2Powq_activation/Const:output:0q_activation/sub:z:0*
T0*
_output_shapes
: k
q_activation/sub_1Subq_activation/Cast_1:y:0q_activation/Pow_2:z:0*
T0*
_output_shapes
: �
q_activation/LessEqual	LessEqual'batch_normalization/batchnorm/add_1:z:0q_activation/sub_1:z:0*
T0*+
_output_shapes
:���������dx
q_activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:���������ds
q_activation/ones_like/ShapeShape'batch_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:a
q_activation/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
q_activation/ones_likeFill%q_activation/ones_like/Shape:output:0%q_activation/ones_like/Const:output:0*
T0*+
_output_shapes
:���������dk
q_activation/sub_2Subq_activation/Cast_1:y:0q_activation/Pow_2:z:0*
T0*
_output_shapes
: �
q_activation/mulMulq_activation/ones_like:output:0q_activation/sub_2:z:0*
T0*+
_output_shapes
:���������d�
q_activation/SelectV2SelectV2q_activation/LessEqual:z:0q_activation/Relu:activations:0q_activation/mul:z:0*
T0*+
_output_shapes
:���������d�
q_activation/mul_1Mul'batch_normalization/batchnorm/add_1:z:0q_activation/Cast:y:0*
T0*+
_output_shapes
:���������d�
q_activation/truedivRealDivq_activation/mul_1:z:0q_activation/Cast_1:y:0*
T0*+
_output_shapes
:���������dg
q_activation/NegNegq_activation/truediv:z:0*
T0*+
_output_shapes
:���������dk
q_activation/RoundRoundq_activation/truediv:z:0*
T0*+
_output_shapes
:���������d}
q_activation/addAddV2q_activation/Neg:y:0q_activation/Round:y:0*
T0*+
_output_shapes
:���������du
q_activation/StopGradientStopGradientq_activation/add:z:0*
T0*+
_output_shapes
:���������d�
q_activation/add_1AddV2q_activation/truediv:z:0"q_activation/StopGradient:output:0*
T0*+
_output_shapes
:���������d�
q_activation/truediv_1RealDivq_activation/add_1:z:0q_activation/Cast:y:0*
T0*+
_output_shapes
:���������d]
q_activation/truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?|
q_activation/truediv_2RealDiv!q_activation/truediv_2/x:output:0q_activation/Cast:y:0*
T0*
_output_shapes
: Y
q_activation/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?u
q_activation/sub_3Subq_activation/sub_3/x:output:0q_activation/truediv_2:z:0*
T0*
_output_shapes
: �
"q_activation/clip_by_value/MinimumMinimumq_activation/truediv_1:z:0q_activation/sub_3:z:0*
T0*+
_output_shapes
:���������da
q_activation/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
q_activation/clip_by_valueMaximum&q_activation/clip_by_value/Minimum:z:0%q_activation/clip_by_value/y:output:0*
T0*+
_output_shapes
:���������d�
q_activation/mul_2Mulq_activation/Cast_1:y:0q_activation/clip_by_value:z:0*
T0*+
_output_shapes
:���������do
q_activation/Neg_1Negq_activation/SelectV2:output:0*
T0*+
_output_shapes
:���������d�
q_activation/add_2AddV2q_activation/Neg_1:y:0q_activation/mul_2:z:0*
T0*+
_output_shapes
:���������dY
q_activation/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
q_activation/mul_3Mulq_activation/mul_3/x:output:0q_activation/add_2:z:0*
T0*+
_output_shapes
:���������dy
q_activation/StopGradient_1StopGradientq_activation/mul_3:z:0*
T0*+
_output_shapes
:���������d�
q_activation/add_3AddV2q_activation/SelectV2:output:0$q_activation/StopGradient_1:output:0*
T0*+
_output_shapes
:���������dQ
q_dense_1/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :Q
q_dense_1/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :i
q_dense_1/PowPowq_dense_1/Pow/x:output:0q_dense_1/Pow/y:output:0*
T0*
_output_shapes
: Y
q_dense_1/CastCastq_dense_1/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: z
q_dense_1/ReadVariableOpReadVariableOp!q_dense_1_readvariableop_resource*
_output_shapes

:$*
dtype0T
q_dense_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Cy
q_dense_1/mulMul q_dense_1/ReadVariableOp:value:0q_dense_1/mul/y:output:0*
T0*
_output_shapes

:$l
q_dense_1/truedivRealDivq_dense_1/mul:z:0q_dense_1/Cast:y:0*
T0*
_output_shapes

:$T
q_dense_1/NegNegq_dense_1/truediv:z:0*
T0*
_output_shapes

:$X
q_dense_1/RoundRoundq_dense_1/truediv:z:0*
T0*
_output_shapes

:$g
q_dense_1/addAddV2q_dense_1/Neg:y:0q_dense_1/Round:y:0*
T0*
_output_shapes

:$b
q_dense_1/StopGradientStopGradientq_dense_1/add:z:0*
T0*
_output_shapes

:$y
q_dense_1/add_1AddV2q_dense_1/truediv:z:0q_dense_1/StopGradient:output:0*
T0*
_output_shapes

:$f
!q_dense_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B�
q_dense_1/clip_by_value/MinimumMinimumq_dense_1/add_1:z:0*q_dense_1/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:$^
q_dense_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
q_dense_1/clip_by_valueMaximum#q_dense_1/clip_by_value/Minimum:z:0"q_dense_1/clip_by_value/y:output:0*
T0*
_output_shapes

:$p
q_dense_1/mul_1Mulq_dense_1/Cast:y:0q_dense_1/clip_by_value:z:0*
T0*
_output_shapes

:$Z
q_dense_1/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C|
q_dense_1/truediv_1RealDivq_dense_1/mul_1:z:0q_dense_1/truediv_1/y:output:0*
T0*
_output_shapes

:$V
q_dense_1/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
q_dense_1/mul_2Mulq_dense_1/mul_2/x:output:0q_dense_1/truediv_1:z:0*
T0*
_output_shapes

:$|
q_dense_1/ReadVariableOp_1ReadVariableOp!q_dense_1_readvariableop_resource*
_output_shapes

:$*
dtype0c
q_dense_1/Neg_1Neg"q_dense_1/ReadVariableOp_1:value:0*
T0*
_output_shapes

:$k
q_dense_1/add_2AddV2q_dense_1/Neg_1:y:0q_dense_1/mul_2:z:0*
T0*
_output_shapes

:$V
q_dense_1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?p
q_dense_1/mul_3Mulq_dense_1/mul_3/x:output:0q_dense_1/add_2:z:0*
T0*
_output_shapes

:$f
q_dense_1/StopGradient_1StopGradientq_dense_1/mul_3:z:0*
T0*
_output_shapes

:$|
q_dense_1/ReadVariableOp_2ReadVariableOp!q_dense_1_readvariableop_resource*
_output_shapes

:$*
dtype0�
q_dense_1/add_3AddV2"q_dense_1/ReadVariableOp_2:value:0!q_dense_1/StopGradient_1:output:0*
T0*
_output_shapes

:$U
q_dense_1/ShapeShapeq_activation/add_3:z:0*
T0*
_output_shapes
:e
q_dense_1/unstackUnpackq_dense_1/Shape:output:0*
T0*
_output_shapes
: : : *	
numb
q_dense_1/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"   $   g
q_dense_1/unstack_1Unpackq_dense_1/Shape_1:output:0*
T0*
_output_shapes
: : *	
numh
q_dense_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
q_dense_1/ReshapeReshapeq_activation/add_3:z:0 q_dense_1/Reshape/shape:output:0*
T0*'
_output_shapes
:���������i
q_dense_1/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
q_dense_1/transpose	Transposeq_dense_1/add_3:z:0!q_dense_1/transpose/perm:output:0*
T0*
_output_shapes

:$j
q_dense_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   �����
q_dense_1/Reshape_1Reshapeq_dense_1/transpose:y:0"q_dense_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:$�
q_dense_1/MatMulMatMulq_dense_1/Reshape:output:0q_dense_1/Reshape_1:output:0*
T0*'
_output_shapes
:���������$]
q_dense_1/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d]
q_dense_1/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :$�
q_dense_1/Reshape_2/shapePackq_dense_1/unstack:output:0$q_dense_1/Reshape_2/shape/1:output:0$q_dense_1/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:�
q_dense_1/Reshape_2Reshapeq_dense_1/MatMul:product:0"q_dense_1/Reshape_2/shape:output:0*
T0*+
_output_shapes
:���������d$S
q_dense_1/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :S
q_dense_1/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :o
q_dense_1/Pow_1Powq_dense_1/Pow_1/x:output:0q_dense_1/Pow_1/y:output:0*
T0*
_output_shapes
: ]
q_dense_1/Cast_1Castq_dense_1/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: z
q_dense_1/ReadVariableOp_3ReadVariableOp#q_dense_1_readvariableop_3_resource*
_output_shapes
:$*
dtype0V
q_dense_1/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C{
q_dense_1/mul_4Mul"q_dense_1/ReadVariableOp_3:value:0q_dense_1/mul_4/y:output:0*
T0*
_output_shapes
:$n
q_dense_1/truediv_2RealDivq_dense_1/mul_4:z:0q_dense_1/Cast_1:y:0*
T0*
_output_shapes
:$T
q_dense_1/Neg_2Negq_dense_1/truediv_2:z:0*
T0*
_output_shapes
:$X
q_dense_1/Round_1Roundq_dense_1/truediv_2:z:0*
T0*
_output_shapes
:$i
q_dense_1/add_4AddV2q_dense_1/Neg_2:y:0q_dense_1/Round_1:y:0*
T0*
_output_shapes
:$b
q_dense_1/StopGradient_2StopGradientq_dense_1/add_4:z:0*
T0*
_output_shapes
:$y
q_dense_1/add_5AddV2q_dense_1/truediv_2:z:0!q_dense_1/StopGradient_2:output:0*
T0*
_output_shapes
:$h
#q_dense_1/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B�
!q_dense_1/clip_by_value_1/MinimumMinimumq_dense_1/add_5:z:0,q_dense_1/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:$`
q_dense_1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
q_dense_1/clip_by_value_1Maximum%q_dense_1/clip_by_value_1/Minimum:z:0$q_dense_1/clip_by_value_1/y:output:0*
T0*
_output_shapes
:$p
q_dense_1/mul_5Mulq_dense_1/Cast_1:y:0q_dense_1/clip_by_value_1:z:0*
T0*
_output_shapes
:$Z
q_dense_1/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Cx
q_dense_1/truediv_3RealDivq_dense_1/mul_5:z:0q_dense_1/truediv_3/y:output:0*
T0*
_output_shapes
:$V
q_dense_1/mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?p
q_dense_1/mul_6Mulq_dense_1/mul_6/x:output:0q_dense_1/truediv_3:z:0*
T0*
_output_shapes
:$z
q_dense_1/ReadVariableOp_4ReadVariableOp#q_dense_1_readvariableop_3_resource*
_output_shapes
:$*
dtype0_
q_dense_1/Neg_3Neg"q_dense_1/ReadVariableOp_4:value:0*
T0*
_output_shapes
:$g
q_dense_1/add_6AddV2q_dense_1/Neg_3:y:0q_dense_1/mul_6:z:0*
T0*
_output_shapes
:$V
q_dense_1/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?l
q_dense_1/mul_7Mulq_dense_1/mul_7/x:output:0q_dense_1/add_6:z:0*
T0*
_output_shapes
:$b
q_dense_1/StopGradient_3StopGradientq_dense_1/mul_7:z:0*
T0*
_output_shapes
:$z
q_dense_1/ReadVariableOp_5ReadVariableOp#q_dense_1_readvariableop_3_resource*
_output_shapes
:$*
dtype0�
q_dense_1/add_7AddV2"q_dense_1/ReadVariableOp_5:value:0!q_dense_1/StopGradient_3:output:0*
T0*
_output_shapes
:$�
q_dense_1/BiasAddBiasAddq_dense_1/Reshape_2:output:0q_dense_1/add_7:z:0*
T0*+
_output_shapes
:���������d$�
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
"batch_normalization_1/moments/meanMeanq_dense_1/BiasAdd:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:$*
	keep_dims(�
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*"
_output_shapes
:$�
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferenceq_dense_1/BiasAdd:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������d$�
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:$*
	keep_dims(�
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:$*
squeeze_dims
 �
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes
:$*
squeeze_dims
 p
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *��L=�
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes
:$*
dtype0�
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes
:$�
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:$�
%batch_normalization_1/AssignMovingAvgAssignSubVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *��L=�
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes
:$*
dtype0�
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:$�
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:$�
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:$|
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:$�
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:$*
dtype0�
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:$�
%batch_normalization_1/batchnorm/mul_1Mulq_dense_1/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������d$�
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:$�
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:$*
dtype0�
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:$�
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������d$V
q_activation_1/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :V
q_activation_1/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :x
q_activation_1/PowPowq_activation_1/Pow/x:output:0q_activation_1/Pow/y:output:0*
T0*
_output_shapes
: c
q_activation_1/CastCastq_activation_1/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: X
q_activation_1/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :X
q_activation_1/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :~
q_activation_1/Pow_1Powq_activation_1/Pow_1/x:output:0q_activation_1/Pow_1/y:output:0*
T0*
_output_shapes
: g
q_activation_1/Cast_1Castq_activation_1/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: Y
q_activation_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
q_activation_1/Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B :o
q_activation_1/Cast_2Cast q_activation_1/Cast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: Y
q_activation_1/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   At
q_activation_1/subSubq_activation_1/Cast_2:y:0q_activation_1/sub/y:output:0*
T0*
_output_shapes
: s
q_activation_1/Pow_2Powq_activation_1/Const:output:0q_activation_1/sub:z:0*
T0*
_output_shapes
: q
q_activation_1/sub_1Subq_activation_1/Cast_1:y:0q_activation_1/Pow_2:z:0*
T0*
_output_shapes
: �
q_activation_1/LessEqual	LessEqual)batch_normalization_1/batchnorm/add_1:z:0q_activation_1/sub_1:z:0*
T0*+
_output_shapes
:���������d$|
q_activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:���������d$w
q_activation_1/ones_like/ShapeShape)batch_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:c
q_activation_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
q_activation_1/ones_likeFill'q_activation_1/ones_like/Shape:output:0'q_activation_1/ones_like/Const:output:0*
T0*+
_output_shapes
:���������d$q
q_activation_1/sub_2Subq_activation_1/Cast_1:y:0q_activation_1/Pow_2:z:0*
T0*
_output_shapes
: �
q_activation_1/mulMul!q_activation_1/ones_like:output:0q_activation_1/sub_2:z:0*
T0*+
_output_shapes
:���������d$�
q_activation_1/SelectV2SelectV2q_activation_1/LessEqual:z:0!q_activation_1/Relu:activations:0q_activation_1/mul:z:0*
T0*+
_output_shapes
:���������d$�
q_activation_1/mul_1Mul)batch_normalization_1/batchnorm/add_1:z:0q_activation_1/Cast:y:0*
T0*+
_output_shapes
:���������d$�
q_activation_1/truedivRealDivq_activation_1/mul_1:z:0q_activation_1/Cast_1:y:0*
T0*+
_output_shapes
:���������d$k
q_activation_1/NegNegq_activation_1/truediv:z:0*
T0*+
_output_shapes
:���������d$o
q_activation_1/RoundRoundq_activation_1/truediv:z:0*
T0*+
_output_shapes
:���������d$�
q_activation_1/addAddV2q_activation_1/Neg:y:0q_activation_1/Round:y:0*
T0*+
_output_shapes
:���������d$y
q_activation_1/StopGradientStopGradientq_activation_1/add:z:0*
T0*+
_output_shapes
:���������d$�
q_activation_1/add_1AddV2q_activation_1/truediv:z:0$q_activation_1/StopGradient:output:0*
T0*+
_output_shapes
:���������d$�
q_activation_1/truediv_1RealDivq_activation_1/add_1:z:0q_activation_1/Cast:y:0*
T0*+
_output_shapes
:���������d$_
q_activation_1/truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
q_activation_1/truediv_2RealDiv#q_activation_1/truediv_2/x:output:0q_activation_1/Cast:y:0*
T0*
_output_shapes
: [
q_activation_1/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?{
q_activation_1/sub_3Subq_activation_1/sub_3/x:output:0q_activation_1/truediv_2:z:0*
T0*
_output_shapes
: �
$q_activation_1/clip_by_value/MinimumMinimumq_activation_1/truediv_1:z:0q_activation_1/sub_3:z:0*
T0*+
_output_shapes
:���������d$c
q_activation_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
q_activation_1/clip_by_valueMaximum(q_activation_1/clip_by_value/Minimum:z:0'q_activation_1/clip_by_value/y:output:0*
T0*+
_output_shapes
:���������d$�
q_activation_1/mul_2Mulq_activation_1/Cast_1:y:0 q_activation_1/clip_by_value:z:0*
T0*+
_output_shapes
:���������d$s
q_activation_1/Neg_1Neg q_activation_1/SelectV2:output:0*
T0*+
_output_shapes
:���������d$�
q_activation_1/add_2AddV2q_activation_1/Neg_1:y:0q_activation_1/mul_2:z:0*
T0*+
_output_shapes
:���������d$[
q_activation_1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
q_activation_1/mul_3Mulq_activation_1/mul_3/x:output:0q_activation_1/add_2:z:0*
T0*+
_output_shapes
:���������d$}
q_activation_1/StopGradient_1StopGradientq_activation_1/mul_3:z:0*
T0*+
_output_shapes
:���������d$�
q_activation_1/add_3AddV2 q_activation_1/SelectV2:output:0&q_activation_1/StopGradient_1:output:0*
T0*+
_output_shapes
:���������d$R
met_weight/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :R
met_weight/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :l
met_weight/PowPowmet_weight/Pow/x:output:0met_weight/Pow/y:output:0*
T0*
_output_shapes
: [
met_weight/CastCastmet_weight/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: |
met_weight/ReadVariableOpReadVariableOp"met_weight_readvariableop_resource*
_output_shapes

:$*
dtype0U
met_weight/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C|
met_weight/mulMul!met_weight/ReadVariableOp:value:0met_weight/mul/y:output:0*
T0*
_output_shapes

:$o
met_weight/truedivRealDivmet_weight/mul:z:0met_weight/Cast:y:0*
T0*
_output_shapes

:$V
met_weight/NegNegmet_weight/truediv:z:0*
T0*
_output_shapes

:$Z
met_weight/RoundRoundmet_weight/truediv:z:0*
T0*
_output_shapes

:$j
met_weight/addAddV2met_weight/Neg:y:0met_weight/Round:y:0*
T0*
_output_shapes

:$d
met_weight/StopGradientStopGradientmet_weight/add:z:0*
T0*
_output_shapes

:$|
met_weight/add_1AddV2met_weight/truediv:z:0 met_weight/StopGradient:output:0*
T0*
_output_shapes

:$g
"met_weight/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B�
 met_weight/clip_by_value/MinimumMinimummet_weight/add_1:z:0+met_weight/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:$_
met_weight/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
met_weight/clip_by_valueMaximum$met_weight/clip_by_value/Minimum:z:0#met_weight/clip_by_value/y:output:0*
T0*
_output_shapes

:$s
met_weight/mul_1Mulmet_weight/Cast:y:0met_weight/clip_by_value:z:0*
T0*
_output_shapes

:$[
met_weight/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C
met_weight/truediv_1RealDivmet_weight/mul_1:z:0met_weight/truediv_1/y:output:0*
T0*
_output_shapes

:$W
met_weight/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?w
met_weight/mul_2Mulmet_weight/mul_2/x:output:0met_weight/truediv_1:z:0*
T0*
_output_shapes

:$~
met_weight/ReadVariableOp_1ReadVariableOp"met_weight_readvariableop_resource*
_output_shapes

:$*
dtype0e
met_weight/Neg_1Neg#met_weight/ReadVariableOp_1:value:0*
T0*
_output_shapes

:$n
met_weight/add_2AddV2met_weight/Neg_1:y:0met_weight/mul_2:z:0*
T0*
_output_shapes

:$W
met_weight/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?s
met_weight/mul_3Mulmet_weight/mul_3/x:output:0met_weight/add_2:z:0*
T0*
_output_shapes

:$h
met_weight/StopGradient_1StopGradientmet_weight/mul_3:z:0*
T0*
_output_shapes

:$~
met_weight/ReadVariableOp_2ReadVariableOp"met_weight_readvariableop_resource*
_output_shapes

:$*
dtype0�
met_weight/add_3AddV2#met_weight/ReadVariableOp_2:value:0"met_weight/StopGradient_1:output:0*
T0*
_output_shapes

:$X
met_weight/ShapeShapeq_activation_1/add_3:z:0*
T0*
_output_shapes
:g
met_weight/unstackUnpackmet_weight/Shape:output:0*
T0*
_output_shapes
: : : *	
numc
met_weight/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"$      i
met_weight/unstack_1Unpackmet_weight/Shape_1:output:0*
T0*
_output_shapes
: : *	
numi
met_weight/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����$   �
met_weight/ReshapeReshapeq_activation_1/add_3:z:0!met_weight/Reshape/shape:output:0*
T0*'
_output_shapes
:���������$j
met_weight/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
met_weight/transpose	Transposemet_weight/add_3:z:0"met_weight/transpose/perm:output:0*
T0*
_output_shapes

:$k
met_weight/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"$   �����
met_weight/Reshape_1Reshapemet_weight/transpose:y:0#met_weight/Reshape_1/shape:output:0*
T0*
_output_shapes

:$�
met_weight/MatMulMatMulmet_weight/Reshape:output:0met_weight/Reshape_1:output:0*
T0*'
_output_shapes
:���������^
met_weight/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d^
met_weight/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
met_weight/Reshape_2/shapePackmet_weight/unstack:output:0%met_weight/Reshape_2/shape/1:output:0%met_weight/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:�
met_weight/Reshape_2Reshapemet_weight/MatMul:product:0#met_weight/Reshape_2/shape:output:0*
T0*+
_output_shapes
:���������dT
met_weight/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :T
met_weight/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :r
met_weight/Pow_1Powmet_weight/Pow_1/x:output:0met_weight/Pow_1/y:output:0*
T0*
_output_shapes
: _
met_weight/Cast_1Castmet_weight/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: |
met_weight/ReadVariableOp_3ReadVariableOp$met_weight_readvariableop_3_resource*
_output_shapes
:*
dtype0W
met_weight/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C~
met_weight/mul_4Mul#met_weight/ReadVariableOp_3:value:0met_weight/mul_4/y:output:0*
T0*
_output_shapes
:q
met_weight/truediv_2RealDivmet_weight/mul_4:z:0met_weight/Cast_1:y:0*
T0*
_output_shapes
:V
met_weight/Neg_2Negmet_weight/truediv_2:z:0*
T0*
_output_shapes
:Z
met_weight/Round_1Roundmet_weight/truediv_2:z:0*
T0*
_output_shapes
:l
met_weight/add_4AddV2met_weight/Neg_2:y:0met_weight/Round_1:y:0*
T0*
_output_shapes
:d
met_weight/StopGradient_2StopGradientmet_weight/add_4:z:0*
T0*
_output_shapes
:|
met_weight/add_5AddV2met_weight/truediv_2:z:0"met_weight/StopGradient_2:output:0*
T0*
_output_shapes
:i
$met_weight/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B�
"met_weight/clip_by_value_1/MinimumMinimummet_weight/add_5:z:0-met_weight/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:a
met_weight/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
met_weight/clip_by_value_1Maximum&met_weight/clip_by_value_1/Minimum:z:0%met_weight/clip_by_value_1/y:output:0*
T0*
_output_shapes
:s
met_weight/mul_5Mulmet_weight/Cast_1:y:0met_weight/clip_by_value_1:z:0*
T0*
_output_shapes
:[
met_weight/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C{
met_weight/truediv_3RealDivmet_weight/mul_5:z:0met_weight/truediv_3/y:output:0*
T0*
_output_shapes
:W
met_weight/mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?s
met_weight/mul_6Mulmet_weight/mul_6/x:output:0met_weight/truediv_3:z:0*
T0*
_output_shapes
:|
met_weight/ReadVariableOp_4ReadVariableOp$met_weight_readvariableop_3_resource*
_output_shapes
:*
dtype0a
met_weight/Neg_3Neg#met_weight/ReadVariableOp_4:value:0*
T0*
_output_shapes
:j
met_weight/add_6AddV2met_weight/Neg_3:y:0met_weight/mul_6:z:0*
T0*
_output_shapes
:W
met_weight/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?o
met_weight/mul_7Mulmet_weight/mul_7/x:output:0met_weight/add_6:z:0*
T0*
_output_shapes
:d
met_weight/StopGradient_3StopGradientmet_weight/mul_7:z:0*
T0*
_output_shapes
:|
met_weight/ReadVariableOp_5ReadVariableOp$met_weight_readvariableop_3_resource*
_output_shapes
:*
dtype0�
met_weight/add_7AddV2#met_weight/ReadVariableOp_5:value:0"met_weight/StopGradient_3:output:0*
T0*
_output_shapes
:�
met_weight/BiasAddBiasAddmet_weight/Reshape_2:output:0met_weight/add_7:z:0*
T0*+
_output_shapes
:���������d�
-met_weight_minus_one/batchnorm/ReadVariableOpReadVariableOp6met_weight_minus_one_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0i
$met_weight_minus_one/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
"met_weight_minus_one/batchnorm/addAddV25met_weight_minus_one/batchnorm/ReadVariableOp:value:0-met_weight_minus_one/batchnorm/add/y:output:0*
T0*
_output_shapes
:z
$met_weight_minus_one/batchnorm/RsqrtRsqrt&met_weight_minus_one/batchnorm/add:z:0*
T0*
_output_shapes
:�
1met_weight_minus_one/batchnorm/mul/ReadVariableOpReadVariableOp:met_weight_minus_one_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
"met_weight_minus_one/batchnorm/mulMul(met_weight_minus_one/batchnorm/Rsqrt:y:09met_weight_minus_one/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
$met_weight_minus_one/batchnorm/mul_1Mulmet_weight/BiasAdd:output:0&met_weight_minus_one/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������d�
/met_weight_minus_one/batchnorm/ReadVariableOp_1ReadVariableOp8met_weight_minus_one_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
$met_weight_minus_one/batchnorm/mul_2Mul7met_weight_minus_one/batchnorm/ReadVariableOp_1:value:0&met_weight_minus_one/batchnorm/mul:z:0*
T0*
_output_shapes
:�
/met_weight_minus_one/batchnorm/ReadVariableOp_2ReadVariableOp8met_weight_minus_one_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
"met_weight_minus_one/batchnorm/subSub7met_weight_minus_one/batchnorm/ReadVariableOp_2:value:0(met_weight_minus_one/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
$met_weight_minus_one/batchnorm/add_1AddV2(met_weight_minus_one/batchnorm/mul_1:z:0&met_weight_minus_one/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������d}
multiply/mulMul(met_weight_minus_one/batchnorm/add_1:z:0inputs_1*
T0*+
_output_shapes
:���������d_
output/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
output/MeanMeanmultiply/mul:z:0&output/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������c
IdentityIdentityoutput/Mean:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp^embedding0/embedding_lookup^embedding1/embedding_lookup^met_weight/ReadVariableOp^met_weight/ReadVariableOp_1^met_weight/ReadVariableOp_2^met_weight/ReadVariableOp_3^met_weight/ReadVariableOp_4^met_weight/ReadVariableOp_5.^met_weight_minus_one/batchnorm/ReadVariableOp0^met_weight_minus_one/batchnorm/ReadVariableOp_10^met_weight_minus_one/batchnorm/ReadVariableOp_22^met_weight_minus_one/batchnorm/mul/ReadVariableOp^q_dense/ReadVariableOp^q_dense/ReadVariableOp_1^q_dense/ReadVariableOp_2^q_dense/ReadVariableOp_3^q_dense/ReadVariableOp_4^q_dense/ReadVariableOp_5^q_dense_1/ReadVariableOp^q_dense_1/ReadVariableOp_1^q_dense_1/ReadVariableOp_2^q_dense_1/ReadVariableOp_3^q_dense_1/ReadVariableOp_4^q_dense_1/ReadVariableOp_5*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|:���������d:���������d:���������d:���������d: : : : : : : : : : : : : : : : : : : : 2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2N
%batch_normalization_1/AssignMovingAvg%batch_normalization_1/AssignMovingAvg2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_1/AssignMovingAvg_1'batch_normalization_1/AssignMovingAvg_12p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2:
embedding0/embedding_lookupembedding0/embedding_lookup2:
embedding1/embedding_lookupembedding1/embedding_lookup26
met_weight/ReadVariableOpmet_weight/ReadVariableOp2:
met_weight/ReadVariableOp_1met_weight/ReadVariableOp_12:
met_weight/ReadVariableOp_2met_weight/ReadVariableOp_22:
met_weight/ReadVariableOp_3met_weight/ReadVariableOp_32:
met_weight/ReadVariableOp_4met_weight/ReadVariableOp_42:
met_weight/ReadVariableOp_5met_weight/ReadVariableOp_52^
-met_weight_minus_one/batchnorm/ReadVariableOp-met_weight_minus_one/batchnorm/ReadVariableOp2b
/met_weight_minus_one/batchnorm/ReadVariableOp_1/met_weight_minus_one/batchnorm/ReadVariableOp_12b
/met_weight_minus_one/batchnorm/ReadVariableOp_2/met_weight_minus_one/batchnorm/ReadVariableOp_22f
1met_weight_minus_one/batchnorm/mul/ReadVariableOp1met_weight_minus_one/batchnorm/mul/ReadVariableOp20
q_dense/ReadVariableOpq_dense/ReadVariableOp24
q_dense/ReadVariableOp_1q_dense/ReadVariableOp_124
q_dense/ReadVariableOp_2q_dense/ReadVariableOp_224
q_dense/ReadVariableOp_3q_dense/ReadVariableOp_324
q_dense/ReadVariableOp_4q_dense/ReadVariableOp_424
q_dense/ReadVariableOp_5q_dense/ReadVariableOp_524
q_dense_1/ReadVariableOpq_dense_1/ReadVariableOp28
q_dense_1/ReadVariableOp_1q_dense_1/ReadVariableOp_128
q_dense_1/ReadVariableOp_2q_dense_1/ReadVariableOp_228
q_dense_1/ReadVariableOp_3q_dense_1/ReadVariableOp_328
q_dense_1/ReadVariableOp_4q_dense_1/ReadVariableOp_428
q_dense_1/ReadVariableOp_5q_dense_1/ReadVariableOp_5:U Q
+
_output_shapes
:���������d
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������d
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������d
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:���������d
"
_user_specified_name
inputs/3
�
�
,__inference_embedding1_layer_call_fn_4872625

inputs
unknown:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_embedding1_layer_call_and_return_conditional_losses_4870724s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������d: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
��
�
"__inference__wrapped_model_4870442

input_cont

input_pxpy

input_cat0

input_cat1;
)model_embedding0_embedding_lookup_4870042:;
)model_embedding1_embedding_lookup_4870048:7
%model_q_dense_readvariableop_resource:5
'model_q_dense_readvariableop_3_resource:I
;model_batch_normalization_batchnorm_readvariableop_resource:M
?model_batch_normalization_batchnorm_mul_readvariableop_resource:K
=model_batch_normalization_batchnorm_readvariableop_1_resource:K
=model_batch_normalization_batchnorm_readvariableop_2_resource:9
'model_q_dense_1_readvariableop_resource:$7
)model_q_dense_1_readvariableop_3_resource:$K
=model_batch_normalization_1_batchnorm_readvariableop_resource:$O
Amodel_batch_normalization_1_batchnorm_mul_readvariableop_resource:$M
?model_batch_normalization_1_batchnorm_readvariableop_1_resource:$M
?model_batch_normalization_1_batchnorm_readvariableop_2_resource:$:
(model_met_weight_readvariableop_resource:$8
*model_met_weight_readvariableop_3_resource:J
<model_met_weight_minus_one_batchnorm_readvariableop_resource:N
@model_met_weight_minus_one_batchnorm_mul_readvariableop_resource:L
>model_met_weight_minus_one_batchnorm_readvariableop_1_resource:L
>model_met_weight_minus_one_batchnorm_readvariableop_2_resource:
identity��2model/batch_normalization/batchnorm/ReadVariableOp�4model/batch_normalization/batchnorm/ReadVariableOp_1�4model/batch_normalization/batchnorm/ReadVariableOp_2�6model/batch_normalization/batchnorm/mul/ReadVariableOp�4model/batch_normalization_1/batchnorm/ReadVariableOp�6model/batch_normalization_1/batchnorm/ReadVariableOp_1�6model/batch_normalization_1/batchnorm/ReadVariableOp_2�8model/batch_normalization_1/batchnorm/mul/ReadVariableOp�!model/embedding0/embedding_lookup�!model/embedding1/embedding_lookup�model/met_weight/ReadVariableOp�!model/met_weight/ReadVariableOp_1�!model/met_weight/ReadVariableOp_2�!model/met_weight/ReadVariableOp_3�!model/met_weight/ReadVariableOp_4�!model/met_weight/ReadVariableOp_5�3model/met_weight_minus_one/batchnorm/ReadVariableOp�5model/met_weight_minus_one/batchnorm/ReadVariableOp_1�5model/met_weight_minus_one/batchnorm/ReadVariableOp_2�7model/met_weight_minus_one/batchnorm/mul/ReadVariableOp�model/q_dense/ReadVariableOp�model/q_dense/ReadVariableOp_1�model/q_dense/ReadVariableOp_2�model/q_dense/ReadVariableOp_3�model/q_dense/ReadVariableOp_4�model/q_dense/ReadVariableOp_5�model/q_dense_1/ReadVariableOp� model/q_dense_1/ReadVariableOp_1� model/q_dense_1/ReadVariableOp_2� model/q_dense_1/ReadVariableOp_3� model/q_dense_1/ReadVariableOp_4� model/q_dense_1/ReadVariableOp_5j
model/embedding0/CastCast
input_cat0*

DstT0*

SrcT0*'
_output_shapes
:���������d�
!model/embedding0/embedding_lookupResourceGather)model_embedding0_embedding_lookup_4870042model/embedding0/Cast:y:0*
Tindices0*<
_class2
0.loc:@model/embedding0/embedding_lookup/4870042*+
_output_shapes
:���������d*
dtype0�
*model/embedding0/embedding_lookup/IdentityIdentity*model/embedding0/embedding_lookup:output:0*
T0*<
_class2
0.loc:@model/embedding0/embedding_lookup/4870042*+
_output_shapes
:���������d�
,model/embedding0/embedding_lookup/Identity_1Identity3model/embedding0/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������dj
model/embedding1/CastCast
input_cat1*

DstT0*

SrcT0*'
_output_shapes
:���������d�
!model/embedding1/embedding_lookupResourceGather)model_embedding1_embedding_lookup_4870048model/embedding1/Cast:y:0*
Tindices0*<
_class2
0.loc:@model/embedding1/embedding_lookup/4870048*+
_output_shapes
:���������d*
dtype0�
*model/embedding1/embedding_lookup/IdentityIdentity*model/embedding1/embedding_lookup:output:0*
T0*<
_class2
0.loc:@model/embedding1/embedding_lookup/4870048*+
_output_shapes
:���������d�
,model/embedding1/embedding_lookup/Identity_1Identity3model/embedding1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������d_
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/concatenate/concatConcatV25model/embedding0/embedding_lookup/Identity_1:output:05model/embedding1/embedding_lookup/Identity_1:output:0&model/concatenate/concat/axis:output:0*
N*
T0*+
_output_shapes
:���������da
model/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/concatenate_1/concatConcatV2
input_cont!model/concatenate/concat:output:0(model/concatenate_1/concat/axis:output:0*
N*
T0*+
_output_shapes
:���������dU
model/q_dense/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :U
model/q_dense/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :u
model/q_dense/PowPowmodel/q_dense/Pow/x:output:0model/q_dense/Pow/y:output:0*
T0*
_output_shapes
: a
model/q_dense/CastCastmodel/q_dense/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: �
model/q_dense/ReadVariableOpReadVariableOp%model_q_dense_readvariableop_resource*
_output_shapes

:*
dtype0X
model/q_dense/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C�
model/q_dense/mulMul$model/q_dense/ReadVariableOp:value:0model/q_dense/mul/y:output:0*
T0*
_output_shapes

:x
model/q_dense/truedivRealDivmodel/q_dense/mul:z:0model/q_dense/Cast:y:0*
T0*
_output_shapes

:\
model/q_dense/NegNegmodel/q_dense/truediv:z:0*
T0*
_output_shapes

:`
model/q_dense/RoundRoundmodel/q_dense/truediv:z:0*
T0*
_output_shapes

:s
model/q_dense/addAddV2model/q_dense/Neg:y:0model/q_dense/Round:y:0*
T0*
_output_shapes

:j
model/q_dense/StopGradientStopGradientmodel/q_dense/add:z:0*
T0*
_output_shapes

:�
model/q_dense/add_1AddV2model/q_dense/truediv:z:0#model/q_dense/StopGradient:output:0*
T0*
_output_shapes

:j
%model/q_dense/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B�
#model/q_dense/clip_by_value/MinimumMinimummodel/q_dense/add_1:z:0.model/q_dense/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:b
model/q_dense/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
model/q_dense/clip_by_valueMaximum'model/q_dense/clip_by_value/Minimum:z:0&model/q_dense/clip_by_value/y:output:0*
T0*
_output_shapes

:|
model/q_dense/mul_1Mulmodel/q_dense/Cast:y:0model/q_dense/clip_by_value:z:0*
T0*
_output_shapes

:^
model/q_dense/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C�
model/q_dense/truediv_1RealDivmodel/q_dense/mul_1:z:0"model/q_dense/truediv_1/y:output:0*
T0*
_output_shapes

:Z
model/q_dense/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/q_dense/mul_2Mulmodel/q_dense/mul_2/x:output:0model/q_dense/truediv_1:z:0*
T0*
_output_shapes

:�
model/q_dense/ReadVariableOp_1ReadVariableOp%model_q_dense_readvariableop_resource*
_output_shapes

:*
dtype0k
model/q_dense/Neg_1Neg&model/q_dense/ReadVariableOp_1:value:0*
T0*
_output_shapes

:w
model/q_dense/add_2AddV2model/q_dense/Neg_1:y:0model/q_dense/mul_2:z:0*
T0*
_output_shapes

:Z
model/q_dense/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?|
model/q_dense/mul_3Mulmodel/q_dense/mul_3/x:output:0model/q_dense/add_2:z:0*
T0*
_output_shapes

:n
model/q_dense/StopGradient_1StopGradientmodel/q_dense/mul_3:z:0*
T0*
_output_shapes

:�
model/q_dense/ReadVariableOp_2ReadVariableOp%model_q_dense_readvariableop_resource*
_output_shapes

:*
dtype0�
model/q_dense/add_3AddV2&model/q_dense/ReadVariableOp_2:value:0%model/q_dense/StopGradient_1:output:0*
T0*
_output_shapes

:f
model/q_dense/ShapeShape#model/concatenate_1/concat:output:0*
T0*
_output_shapes
:m
model/q_dense/unstackUnpackmodel/q_dense/Shape:output:0*
T0*
_output_shapes
: : : *	
numf
model/q_dense/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"      o
model/q_dense/unstack_1Unpackmodel/q_dense/Shape_1:output:0*
T0*
_output_shapes
: : *	
numl
model/q_dense/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
model/q_dense/ReshapeReshape#model/concatenate_1/concat:output:0$model/q_dense/Reshape/shape:output:0*
T0*'
_output_shapes
:���������m
model/q_dense/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
model/q_dense/transpose	Transposemodel/q_dense/add_3:z:0%model/q_dense/transpose/perm:output:0*
T0*
_output_shapes

:n
model/q_dense/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   �����
model/q_dense/Reshape_1Reshapemodel/q_dense/transpose:y:0&model/q_dense/Reshape_1/shape:output:0*
T0*
_output_shapes

:�
model/q_dense/MatMulMatMulmodel/q_dense/Reshape:output:0 model/q_dense/Reshape_1:output:0*
T0*'
_output_shapes
:���������a
model/q_dense/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :da
model/q_dense/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
model/q_dense/Reshape_2/shapePackmodel/q_dense/unstack:output:0(model/q_dense/Reshape_2/shape/1:output:0(model/q_dense/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:�
model/q_dense/Reshape_2Reshapemodel/q_dense/MatMul:product:0&model/q_dense/Reshape_2/shape:output:0*
T0*+
_output_shapes
:���������dW
model/q_dense/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :W
model/q_dense/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
model/q_dense/Pow_1Powmodel/q_dense/Pow_1/x:output:0model/q_dense/Pow_1/y:output:0*
T0*
_output_shapes
: e
model/q_dense/Cast_1Castmodel/q_dense/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
model/q_dense/ReadVariableOp_3ReadVariableOp'model_q_dense_readvariableop_3_resource*
_output_shapes
:*
dtype0Z
model/q_dense/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C�
model/q_dense/mul_4Mul&model/q_dense/ReadVariableOp_3:value:0model/q_dense/mul_4/y:output:0*
T0*
_output_shapes
:z
model/q_dense/truediv_2RealDivmodel/q_dense/mul_4:z:0model/q_dense/Cast_1:y:0*
T0*
_output_shapes
:\
model/q_dense/Neg_2Negmodel/q_dense/truediv_2:z:0*
T0*
_output_shapes
:`
model/q_dense/Round_1Roundmodel/q_dense/truediv_2:z:0*
T0*
_output_shapes
:u
model/q_dense/add_4AddV2model/q_dense/Neg_2:y:0model/q_dense/Round_1:y:0*
T0*
_output_shapes
:j
model/q_dense/StopGradient_2StopGradientmodel/q_dense/add_4:z:0*
T0*
_output_shapes
:�
model/q_dense/add_5AddV2model/q_dense/truediv_2:z:0%model/q_dense/StopGradient_2:output:0*
T0*
_output_shapes
:l
'model/q_dense/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B�
%model/q_dense/clip_by_value_1/MinimumMinimummodel/q_dense/add_5:z:00model/q_dense/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:d
model/q_dense/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
model/q_dense/clip_by_value_1Maximum)model/q_dense/clip_by_value_1/Minimum:z:0(model/q_dense/clip_by_value_1/y:output:0*
T0*
_output_shapes
:|
model/q_dense/mul_5Mulmodel/q_dense/Cast_1:y:0!model/q_dense/clip_by_value_1:z:0*
T0*
_output_shapes
:^
model/q_dense/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C�
model/q_dense/truediv_3RealDivmodel/q_dense/mul_5:z:0"model/q_dense/truediv_3/y:output:0*
T0*
_output_shapes
:Z
model/q_dense/mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?|
model/q_dense/mul_6Mulmodel/q_dense/mul_6/x:output:0model/q_dense/truediv_3:z:0*
T0*
_output_shapes
:�
model/q_dense/ReadVariableOp_4ReadVariableOp'model_q_dense_readvariableop_3_resource*
_output_shapes
:*
dtype0g
model/q_dense/Neg_3Neg&model/q_dense/ReadVariableOp_4:value:0*
T0*
_output_shapes
:s
model/q_dense/add_6AddV2model/q_dense/Neg_3:y:0model/q_dense/mul_6:z:0*
T0*
_output_shapes
:Z
model/q_dense/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?x
model/q_dense/mul_7Mulmodel/q_dense/mul_7/x:output:0model/q_dense/add_6:z:0*
T0*
_output_shapes
:j
model/q_dense/StopGradient_3StopGradientmodel/q_dense/mul_7:z:0*
T0*
_output_shapes
:�
model/q_dense/ReadVariableOp_5ReadVariableOp'model_q_dense_readvariableop_3_resource*
_output_shapes
:*
dtype0�
model/q_dense/add_7AddV2&model/q_dense/ReadVariableOp_5:value:0%model/q_dense/StopGradient_3:output:0*
T0*
_output_shapes
:�
model/q_dense/BiasAddBiasAdd model/q_dense/Reshape_2:output:0model/q_dense/add_7:z:0*
T0*+
_output_shapes
:���������d�
2model/batch_normalization/batchnorm/ReadVariableOpReadVariableOp;model_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0n
)model/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
'model/batch_normalization/batchnorm/addAddV2:model/batch_normalization/batchnorm/ReadVariableOp:value:02model/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
)model/batch_normalization/batchnorm/RsqrtRsqrt+model/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:�
6model/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp?model_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
'model/batch_normalization/batchnorm/mulMul-model/batch_normalization/batchnorm/Rsqrt:y:0>model/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
)model/batch_normalization/batchnorm/mul_1Mulmodel/q_dense/BiasAdd:output:0+model/batch_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������d�
4model/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp=model_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
)model/batch_normalization/batchnorm/mul_2Mul<model/batch_normalization/batchnorm/ReadVariableOp_1:value:0+model/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:�
4model/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp=model_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
'model/batch_normalization/batchnorm/subSub<model/batch_normalization/batchnorm/ReadVariableOp_2:value:0-model/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
)model/batch_normalization/batchnorm/add_1AddV2-model/batch_normalization/batchnorm/mul_1:z:0+model/batch_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������dZ
model/q_activation/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :Z
model/q_activation/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :�
model/q_activation/PowPow!model/q_activation/Pow/x:output:0!model/q_activation/Pow/y:output:0*
T0*
_output_shapes
: k
model/q_activation/CastCastmodel/q_activation/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: \
model/q_activation/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :\
model/q_activation/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
model/q_activation/Pow_1Pow#model/q_activation/Pow_1/x:output:0#model/q_activation/Pow_1/y:output:0*
T0*
_output_shapes
: o
model/q_activation/Cast_1Castmodel/q_activation/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: ]
model/q_activation/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @]
model/q_activation/Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B :w
model/q_activation/Cast_2Cast$model/q_activation/Cast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: ]
model/q_activation/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A�
model/q_activation/subSubmodel/q_activation/Cast_2:y:0!model/q_activation/sub/y:output:0*
T0*
_output_shapes
: 
model/q_activation/Pow_2Pow!model/q_activation/Const:output:0model/q_activation/sub:z:0*
T0*
_output_shapes
: }
model/q_activation/sub_1Submodel/q_activation/Cast_1:y:0model/q_activation/Pow_2:z:0*
T0*
_output_shapes
: �
model/q_activation/LessEqual	LessEqual-model/batch_normalization/batchnorm/add_1:z:0model/q_activation/sub_1:z:0*
T0*+
_output_shapes
:���������d�
model/q_activation/ReluRelu-model/batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:���������d
"model/q_activation/ones_like/ShapeShape-model/batch_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:g
"model/q_activation/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/q_activation/ones_likeFill+model/q_activation/ones_like/Shape:output:0+model/q_activation/ones_like/Const:output:0*
T0*+
_output_shapes
:���������d}
model/q_activation/sub_2Submodel/q_activation/Cast_1:y:0model/q_activation/Pow_2:z:0*
T0*
_output_shapes
: �
model/q_activation/mulMul%model/q_activation/ones_like:output:0model/q_activation/sub_2:z:0*
T0*+
_output_shapes
:���������d�
model/q_activation/SelectV2SelectV2 model/q_activation/LessEqual:z:0%model/q_activation/Relu:activations:0model/q_activation/mul:z:0*
T0*+
_output_shapes
:���������d�
model/q_activation/mul_1Mul-model/batch_normalization/batchnorm/add_1:z:0model/q_activation/Cast:y:0*
T0*+
_output_shapes
:���������d�
model/q_activation/truedivRealDivmodel/q_activation/mul_1:z:0model/q_activation/Cast_1:y:0*
T0*+
_output_shapes
:���������ds
model/q_activation/NegNegmodel/q_activation/truediv:z:0*
T0*+
_output_shapes
:���������dw
model/q_activation/RoundRoundmodel/q_activation/truediv:z:0*
T0*+
_output_shapes
:���������d�
model/q_activation/addAddV2model/q_activation/Neg:y:0model/q_activation/Round:y:0*
T0*+
_output_shapes
:���������d�
model/q_activation/StopGradientStopGradientmodel/q_activation/add:z:0*
T0*+
_output_shapes
:���������d�
model/q_activation/add_1AddV2model/q_activation/truediv:z:0(model/q_activation/StopGradient:output:0*
T0*+
_output_shapes
:���������d�
model/q_activation/truediv_1RealDivmodel/q_activation/add_1:z:0model/q_activation/Cast:y:0*
T0*+
_output_shapes
:���������dc
model/q_activation/truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/q_activation/truediv_2RealDiv'model/q_activation/truediv_2/x:output:0model/q_activation/Cast:y:0*
T0*
_output_shapes
: _
model/q_activation/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/q_activation/sub_3Sub#model/q_activation/sub_3/x:output:0 model/q_activation/truediv_2:z:0*
T0*
_output_shapes
: �
(model/q_activation/clip_by_value/MinimumMinimum model/q_activation/truediv_1:z:0model/q_activation/sub_3:z:0*
T0*+
_output_shapes
:���������dg
"model/q_activation/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
 model/q_activation/clip_by_valueMaximum,model/q_activation/clip_by_value/Minimum:z:0+model/q_activation/clip_by_value/y:output:0*
T0*+
_output_shapes
:���������d�
model/q_activation/mul_2Mulmodel/q_activation/Cast_1:y:0$model/q_activation/clip_by_value:z:0*
T0*+
_output_shapes
:���������d{
model/q_activation/Neg_1Neg$model/q_activation/SelectV2:output:0*
T0*+
_output_shapes
:���������d�
model/q_activation/add_2AddV2model/q_activation/Neg_1:y:0model/q_activation/mul_2:z:0*
T0*+
_output_shapes
:���������d_
model/q_activation/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/q_activation/mul_3Mul#model/q_activation/mul_3/x:output:0model/q_activation/add_2:z:0*
T0*+
_output_shapes
:���������d�
!model/q_activation/StopGradient_1StopGradientmodel/q_activation/mul_3:z:0*
T0*+
_output_shapes
:���������d�
model/q_activation/add_3AddV2$model/q_activation/SelectV2:output:0*model/q_activation/StopGradient_1:output:0*
T0*+
_output_shapes
:���������dW
model/q_dense_1/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :W
model/q_dense_1/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :{
model/q_dense_1/PowPowmodel/q_dense_1/Pow/x:output:0model/q_dense_1/Pow/y:output:0*
T0*
_output_shapes
: e
model/q_dense_1/CastCastmodel/q_dense_1/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: �
model/q_dense_1/ReadVariableOpReadVariableOp'model_q_dense_1_readvariableop_resource*
_output_shapes

:$*
dtype0Z
model/q_dense_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C�
model/q_dense_1/mulMul&model/q_dense_1/ReadVariableOp:value:0model/q_dense_1/mul/y:output:0*
T0*
_output_shapes

:$~
model/q_dense_1/truedivRealDivmodel/q_dense_1/mul:z:0model/q_dense_1/Cast:y:0*
T0*
_output_shapes

:$`
model/q_dense_1/NegNegmodel/q_dense_1/truediv:z:0*
T0*
_output_shapes

:$d
model/q_dense_1/RoundRoundmodel/q_dense_1/truediv:z:0*
T0*
_output_shapes

:$y
model/q_dense_1/addAddV2model/q_dense_1/Neg:y:0model/q_dense_1/Round:y:0*
T0*
_output_shapes

:$n
model/q_dense_1/StopGradientStopGradientmodel/q_dense_1/add:z:0*
T0*
_output_shapes

:$�
model/q_dense_1/add_1AddV2model/q_dense_1/truediv:z:0%model/q_dense_1/StopGradient:output:0*
T0*
_output_shapes

:$l
'model/q_dense_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B�
%model/q_dense_1/clip_by_value/MinimumMinimummodel/q_dense_1/add_1:z:00model/q_dense_1/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:$d
model/q_dense_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
model/q_dense_1/clip_by_valueMaximum)model/q_dense_1/clip_by_value/Minimum:z:0(model/q_dense_1/clip_by_value/y:output:0*
T0*
_output_shapes

:$�
model/q_dense_1/mul_1Mulmodel/q_dense_1/Cast:y:0!model/q_dense_1/clip_by_value:z:0*
T0*
_output_shapes

:$`
model/q_dense_1/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C�
model/q_dense_1/truediv_1RealDivmodel/q_dense_1/mul_1:z:0$model/q_dense_1/truediv_1/y:output:0*
T0*
_output_shapes

:$\
model/q_dense_1/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/q_dense_1/mul_2Mul model/q_dense_1/mul_2/x:output:0model/q_dense_1/truediv_1:z:0*
T0*
_output_shapes

:$�
 model/q_dense_1/ReadVariableOp_1ReadVariableOp'model_q_dense_1_readvariableop_resource*
_output_shapes

:$*
dtype0o
model/q_dense_1/Neg_1Neg(model/q_dense_1/ReadVariableOp_1:value:0*
T0*
_output_shapes

:$}
model/q_dense_1/add_2AddV2model/q_dense_1/Neg_1:y:0model/q_dense_1/mul_2:z:0*
T0*
_output_shapes

:$\
model/q_dense_1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/q_dense_1/mul_3Mul model/q_dense_1/mul_3/x:output:0model/q_dense_1/add_2:z:0*
T0*
_output_shapes

:$r
model/q_dense_1/StopGradient_1StopGradientmodel/q_dense_1/mul_3:z:0*
T0*
_output_shapes

:$�
 model/q_dense_1/ReadVariableOp_2ReadVariableOp'model_q_dense_1_readvariableop_resource*
_output_shapes

:$*
dtype0�
model/q_dense_1/add_3AddV2(model/q_dense_1/ReadVariableOp_2:value:0'model/q_dense_1/StopGradient_1:output:0*
T0*
_output_shapes

:$a
model/q_dense_1/ShapeShapemodel/q_activation/add_3:z:0*
T0*
_output_shapes
:q
model/q_dense_1/unstackUnpackmodel/q_dense_1/Shape:output:0*
T0*
_output_shapes
: : : *	
numh
model/q_dense_1/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"   $   s
model/q_dense_1/unstack_1Unpack model/q_dense_1/Shape_1:output:0*
T0*
_output_shapes
: : *	
numn
model/q_dense_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
model/q_dense_1/ReshapeReshapemodel/q_activation/add_3:z:0&model/q_dense_1/Reshape/shape:output:0*
T0*'
_output_shapes
:���������o
model/q_dense_1/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
model/q_dense_1/transpose	Transposemodel/q_dense_1/add_3:z:0'model/q_dense_1/transpose/perm:output:0*
T0*
_output_shapes

:$p
model/q_dense_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   �����
model/q_dense_1/Reshape_1Reshapemodel/q_dense_1/transpose:y:0(model/q_dense_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:$�
model/q_dense_1/MatMulMatMul model/q_dense_1/Reshape:output:0"model/q_dense_1/Reshape_1:output:0*
T0*'
_output_shapes
:���������$c
!model/q_dense_1/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :dc
!model/q_dense_1/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :$�
model/q_dense_1/Reshape_2/shapePack model/q_dense_1/unstack:output:0*model/q_dense_1/Reshape_2/shape/1:output:0*model/q_dense_1/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:�
model/q_dense_1/Reshape_2Reshape model/q_dense_1/MatMul:product:0(model/q_dense_1/Reshape_2/shape:output:0*
T0*+
_output_shapes
:���������d$Y
model/q_dense_1/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
model/q_dense_1/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
model/q_dense_1/Pow_1Pow model/q_dense_1/Pow_1/x:output:0 model/q_dense_1/Pow_1/y:output:0*
T0*
_output_shapes
: i
model/q_dense_1/Cast_1Castmodel/q_dense_1/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
 model/q_dense_1/ReadVariableOp_3ReadVariableOp)model_q_dense_1_readvariableop_3_resource*
_output_shapes
:$*
dtype0\
model/q_dense_1/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C�
model/q_dense_1/mul_4Mul(model/q_dense_1/ReadVariableOp_3:value:0 model/q_dense_1/mul_4/y:output:0*
T0*
_output_shapes
:$�
model/q_dense_1/truediv_2RealDivmodel/q_dense_1/mul_4:z:0model/q_dense_1/Cast_1:y:0*
T0*
_output_shapes
:$`
model/q_dense_1/Neg_2Negmodel/q_dense_1/truediv_2:z:0*
T0*
_output_shapes
:$d
model/q_dense_1/Round_1Roundmodel/q_dense_1/truediv_2:z:0*
T0*
_output_shapes
:${
model/q_dense_1/add_4AddV2model/q_dense_1/Neg_2:y:0model/q_dense_1/Round_1:y:0*
T0*
_output_shapes
:$n
model/q_dense_1/StopGradient_2StopGradientmodel/q_dense_1/add_4:z:0*
T0*
_output_shapes
:$�
model/q_dense_1/add_5AddV2model/q_dense_1/truediv_2:z:0'model/q_dense_1/StopGradient_2:output:0*
T0*
_output_shapes
:$n
)model/q_dense_1/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B�
'model/q_dense_1/clip_by_value_1/MinimumMinimummodel/q_dense_1/add_5:z:02model/q_dense_1/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:$f
!model/q_dense_1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
model/q_dense_1/clip_by_value_1Maximum+model/q_dense_1/clip_by_value_1/Minimum:z:0*model/q_dense_1/clip_by_value_1/y:output:0*
T0*
_output_shapes
:$�
model/q_dense_1/mul_5Mulmodel/q_dense_1/Cast_1:y:0#model/q_dense_1/clip_by_value_1:z:0*
T0*
_output_shapes
:$`
model/q_dense_1/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C�
model/q_dense_1/truediv_3RealDivmodel/q_dense_1/mul_5:z:0$model/q_dense_1/truediv_3/y:output:0*
T0*
_output_shapes
:$\
model/q_dense_1/mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/q_dense_1/mul_6Mul model/q_dense_1/mul_6/x:output:0model/q_dense_1/truediv_3:z:0*
T0*
_output_shapes
:$�
 model/q_dense_1/ReadVariableOp_4ReadVariableOp)model_q_dense_1_readvariableop_3_resource*
_output_shapes
:$*
dtype0k
model/q_dense_1/Neg_3Neg(model/q_dense_1/ReadVariableOp_4:value:0*
T0*
_output_shapes
:$y
model/q_dense_1/add_6AddV2model/q_dense_1/Neg_3:y:0model/q_dense_1/mul_6:z:0*
T0*
_output_shapes
:$\
model/q_dense_1/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?~
model/q_dense_1/mul_7Mul model/q_dense_1/mul_7/x:output:0model/q_dense_1/add_6:z:0*
T0*
_output_shapes
:$n
model/q_dense_1/StopGradient_3StopGradientmodel/q_dense_1/mul_7:z:0*
T0*
_output_shapes
:$�
 model/q_dense_1/ReadVariableOp_5ReadVariableOp)model_q_dense_1_readvariableop_3_resource*
_output_shapes
:$*
dtype0�
model/q_dense_1/add_7AddV2(model/q_dense_1/ReadVariableOp_5:value:0'model/q_dense_1/StopGradient_3:output:0*
T0*
_output_shapes
:$�
model/q_dense_1/BiasAddBiasAdd"model/q_dense_1/Reshape_2:output:0model/q_dense_1/add_7:z:0*
T0*+
_output_shapes
:���������d$�
4model/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp=model_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:$*
dtype0p
+model/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
)model/batch_normalization_1/batchnorm/addAddV2<model/batch_normalization_1/batchnorm/ReadVariableOp:value:04model/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:$�
+model/batch_normalization_1/batchnorm/RsqrtRsqrt-model/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:$�
8model/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:$*
dtype0�
)model/batch_normalization_1/batchnorm/mulMul/model/batch_normalization_1/batchnorm/Rsqrt:y:0@model/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:$�
+model/batch_normalization_1/batchnorm/mul_1Mul model/q_dense_1/BiasAdd:output:0-model/batch_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������d$�
6model/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp?model_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:$*
dtype0�
+model/batch_normalization_1/batchnorm/mul_2Mul>model/batch_normalization_1/batchnorm/ReadVariableOp_1:value:0-model/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:$�
6model/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp?model_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:$*
dtype0�
)model/batch_normalization_1/batchnorm/subSub>model/batch_normalization_1/batchnorm/ReadVariableOp_2:value:0/model/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:$�
+model/batch_normalization_1/batchnorm/add_1AddV2/model/batch_normalization_1/batchnorm/mul_1:z:0-model/batch_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������d$\
model/q_activation_1/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :\
model/q_activation_1/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :�
model/q_activation_1/PowPow#model/q_activation_1/Pow/x:output:0#model/q_activation_1/Pow/y:output:0*
T0*
_output_shapes
: o
model/q_activation_1/CastCastmodel/q_activation_1/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: ^
model/q_activation_1/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :^
model/q_activation_1/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
model/q_activation_1/Pow_1Pow%model/q_activation_1/Pow_1/x:output:0%model/q_activation_1/Pow_1/y:output:0*
T0*
_output_shapes
: s
model/q_activation_1/Cast_1Castmodel/q_activation_1/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: _
model/q_activation_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @_
model/q_activation_1/Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B :{
model/q_activation_1/Cast_2Cast&model/q_activation_1/Cast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: _
model/q_activation_1/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   A�
model/q_activation_1/subSubmodel/q_activation_1/Cast_2:y:0#model/q_activation_1/sub/y:output:0*
T0*
_output_shapes
: �
model/q_activation_1/Pow_2Pow#model/q_activation_1/Const:output:0model/q_activation_1/sub:z:0*
T0*
_output_shapes
: �
model/q_activation_1/sub_1Submodel/q_activation_1/Cast_1:y:0model/q_activation_1/Pow_2:z:0*
T0*
_output_shapes
: �
model/q_activation_1/LessEqual	LessEqual/model/batch_normalization_1/batchnorm/add_1:z:0model/q_activation_1/sub_1:z:0*
T0*+
_output_shapes
:���������d$�
model/q_activation_1/ReluRelu/model/batch_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:���������d$�
$model/q_activation_1/ones_like/ShapeShape/model/batch_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:i
$model/q_activation_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/q_activation_1/ones_likeFill-model/q_activation_1/ones_like/Shape:output:0-model/q_activation_1/ones_like/Const:output:0*
T0*+
_output_shapes
:���������d$�
model/q_activation_1/sub_2Submodel/q_activation_1/Cast_1:y:0model/q_activation_1/Pow_2:z:0*
T0*
_output_shapes
: �
model/q_activation_1/mulMul'model/q_activation_1/ones_like:output:0model/q_activation_1/sub_2:z:0*
T0*+
_output_shapes
:���������d$�
model/q_activation_1/SelectV2SelectV2"model/q_activation_1/LessEqual:z:0'model/q_activation_1/Relu:activations:0model/q_activation_1/mul:z:0*
T0*+
_output_shapes
:���������d$�
model/q_activation_1/mul_1Mul/model/batch_normalization_1/batchnorm/add_1:z:0model/q_activation_1/Cast:y:0*
T0*+
_output_shapes
:���������d$�
model/q_activation_1/truedivRealDivmodel/q_activation_1/mul_1:z:0model/q_activation_1/Cast_1:y:0*
T0*+
_output_shapes
:���������d$w
model/q_activation_1/NegNeg model/q_activation_1/truediv:z:0*
T0*+
_output_shapes
:���������d${
model/q_activation_1/RoundRound model/q_activation_1/truediv:z:0*
T0*+
_output_shapes
:���������d$�
model/q_activation_1/addAddV2model/q_activation_1/Neg:y:0model/q_activation_1/Round:y:0*
T0*+
_output_shapes
:���������d$�
!model/q_activation_1/StopGradientStopGradientmodel/q_activation_1/add:z:0*
T0*+
_output_shapes
:���������d$�
model/q_activation_1/add_1AddV2 model/q_activation_1/truediv:z:0*model/q_activation_1/StopGradient:output:0*
T0*+
_output_shapes
:���������d$�
model/q_activation_1/truediv_1RealDivmodel/q_activation_1/add_1:z:0model/q_activation_1/Cast:y:0*
T0*+
_output_shapes
:���������d$e
 model/q_activation_1/truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/q_activation_1/truediv_2RealDiv)model/q_activation_1/truediv_2/x:output:0model/q_activation_1/Cast:y:0*
T0*
_output_shapes
: a
model/q_activation_1/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/q_activation_1/sub_3Sub%model/q_activation_1/sub_3/x:output:0"model/q_activation_1/truediv_2:z:0*
T0*
_output_shapes
: �
*model/q_activation_1/clip_by_value/MinimumMinimum"model/q_activation_1/truediv_1:z:0model/q_activation_1/sub_3:z:0*
T0*+
_output_shapes
:���������d$i
$model/q_activation_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
"model/q_activation_1/clip_by_valueMaximum.model/q_activation_1/clip_by_value/Minimum:z:0-model/q_activation_1/clip_by_value/y:output:0*
T0*+
_output_shapes
:���������d$�
model/q_activation_1/mul_2Mulmodel/q_activation_1/Cast_1:y:0&model/q_activation_1/clip_by_value:z:0*
T0*+
_output_shapes
:���������d$
model/q_activation_1/Neg_1Neg&model/q_activation_1/SelectV2:output:0*
T0*+
_output_shapes
:���������d$�
model/q_activation_1/add_2AddV2model/q_activation_1/Neg_1:y:0model/q_activation_1/mul_2:z:0*
T0*+
_output_shapes
:���������d$a
model/q_activation_1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/q_activation_1/mul_3Mul%model/q_activation_1/mul_3/x:output:0model/q_activation_1/add_2:z:0*
T0*+
_output_shapes
:���������d$�
#model/q_activation_1/StopGradient_1StopGradientmodel/q_activation_1/mul_3:z:0*
T0*+
_output_shapes
:���������d$�
model/q_activation_1/add_3AddV2&model/q_activation_1/SelectV2:output:0,model/q_activation_1/StopGradient_1:output:0*
T0*+
_output_shapes
:���������d$X
model/met_weight/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :X
model/met_weight/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :~
model/met_weight/PowPowmodel/met_weight/Pow/x:output:0model/met_weight/Pow/y:output:0*
T0*
_output_shapes
: g
model/met_weight/CastCastmodel/met_weight/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: �
model/met_weight/ReadVariableOpReadVariableOp(model_met_weight_readvariableop_resource*
_output_shapes

:$*
dtype0[
model/met_weight/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C�
model/met_weight/mulMul'model/met_weight/ReadVariableOp:value:0model/met_weight/mul/y:output:0*
T0*
_output_shapes

:$�
model/met_weight/truedivRealDivmodel/met_weight/mul:z:0model/met_weight/Cast:y:0*
T0*
_output_shapes

:$b
model/met_weight/NegNegmodel/met_weight/truediv:z:0*
T0*
_output_shapes

:$f
model/met_weight/RoundRoundmodel/met_weight/truediv:z:0*
T0*
_output_shapes

:$|
model/met_weight/addAddV2model/met_weight/Neg:y:0model/met_weight/Round:y:0*
T0*
_output_shapes

:$p
model/met_weight/StopGradientStopGradientmodel/met_weight/add:z:0*
T0*
_output_shapes

:$�
model/met_weight/add_1AddV2model/met_weight/truediv:z:0&model/met_weight/StopGradient:output:0*
T0*
_output_shapes

:$m
(model/met_weight/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B�
&model/met_weight/clip_by_value/MinimumMinimummodel/met_weight/add_1:z:01model/met_weight/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:$e
 model/met_weight/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
model/met_weight/clip_by_valueMaximum*model/met_weight/clip_by_value/Minimum:z:0)model/met_weight/clip_by_value/y:output:0*
T0*
_output_shapes

:$�
model/met_weight/mul_1Mulmodel/met_weight/Cast:y:0"model/met_weight/clip_by_value:z:0*
T0*
_output_shapes

:$a
model/met_weight/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C�
model/met_weight/truediv_1RealDivmodel/met_weight/mul_1:z:0%model/met_weight/truediv_1/y:output:0*
T0*
_output_shapes

:$]
model/met_weight/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/met_weight/mul_2Mul!model/met_weight/mul_2/x:output:0model/met_weight/truediv_1:z:0*
T0*
_output_shapes

:$�
!model/met_weight/ReadVariableOp_1ReadVariableOp(model_met_weight_readvariableop_resource*
_output_shapes

:$*
dtype0q
model/met_weight/Neg_1Neg)model/met_weight/ReadVariableOp_1:value:0*
T0*
_output_shapes

:$�
model/met_weight/add_2AddV2model/met_weight/Neg_1:y:0model/met_weight/mul_2:z:0*
T0*
_output_shapes

:$]
model/met_weight/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/met_weight/mul_3Mul!model/met_weight/mul_3/x:output:0model/met_weight/add_2:z:0*
T0*
_output_shapes

:$t
model/met_weight/StopGradient_1StopGradientmodel/met_weight/mul_3:z:0*
T0*
_output_shapes

:$�
!model/met_weight/ReadVariableOp_2ReadVariableOp(model_met_weight_readvariableop_resource*
_output_shapes

:$*
dtype0�
model/met_weight/add_3AddV2)model/met_weight/ReadVariableOp_2:value:0(model/met_weight/StopGradient_1:output:0*
T0*
_output_shapes

:$d
model/met_weight/ShapeShapemodel/q_activation_1/add_3:z:0*
T0*
_output_shapes
:s
model/met_weight/unstackUnpackmodel/met_weight/Shape:output:0*
T0*
_output_shapes
: : : *	
numi
model/met_weight/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"$      u
model/met_weight/unstack_1Unpack!model/met_weight/Shape_1:output:0*
T0*
_output_shapes
: : *	
numo
model/met_weight/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����$   �
model/met_weight/ReshapeReshapemodel/q_activation_1/add_3:z:0'model/met_weight/Reshape/shape:output:0*
T0*'
_output_shapes
:���������$p
model/met_weight/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
model/met_weight/transpose	Transposemodel/met_weight/add_3:z:0(model/met_weight/transpose/perm:output:0*
T0*
_output_shapes

:$q
 model/met_weight/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"$   �����
model/met_weight/Reshape_1Reshapemodel/met_weight/transpose:y:0)model/met_weight/Reshape_1/shape:output:0*
T0*
_output_shapes

:$�
model/met_weight/MatMulMatMul!model/met_weight/Reshape:output:0#model/met_weight/Reshape_1:output:0*
T0*'
_output_shapes
:���������d
"model/met_weight/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :dd
"model/met_weight/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
 model/met_weight/Reshape_2/shapePack!model/met_weight/unstack:output:0+model/met_weight/Reshape_2/shape/1:output:0+model/met_weight/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:�
model/met_weight/Reshape_2Reshape!model/met_weight/MatMul:product:0)model/met_weight/Reshape_2/shape:output:0*
T0*+
_output_shapes
:���������dZ
model/met_weight/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :Z
model/met_weight/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
model/met_weight/Pow_1Pow!model/met_weight/Pow_1/x:output:0!model/met_weight/Pow_1/y:output:0*
T0*
_output_shapes
: k
model/met_weight/Cast_1Castmodel/met_weight/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: �
!model/met_weight/ReadVariableOp_3ReadVariableOp*model_met_weight_readvariableop_3_resource*
_output_shapes
:*
dtype0]
model/met_weight/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C�
model/met_weight/mul_4Mul)model/met_weight/ReadVariableOp_3:value:0!model/met_weight/mul_4/y:output:0*
T0*
_output_shapes
:�
model/met_weight/truediv_2RealDivmodel/met_weight/mul_4:z:0model/met_weight/Cast_1:y:0*
T0*
_output_shapes
:b
model/met_weight/Neg_2Negmodel/met_weight/truediv_2:z:0*
T0*
_output_shapes
:f
model/met_weight/Round_1Roundmodel/met_weight/truediv_2:z:0*
T0*
_output_shapes
:~
model/met_weight/add_4AddV2model/met_weight/Neg_2:y:0model/met_weight/Round_1:y:0*
T0*
_output_shapes
:p
model/met_weight/StopGradient_2StopGradientmodel/met_weight/add_4:z:0*
T0*
_output_shapes
:�
model/met_weight/add_5AddV2model/met_weight/truediv_2:z:0(model/met_weight/StopGradient_2:output:0*
T0*
_output_shapes
:o
*model/met_weight/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B�
(model/met_weight/clip_by_value_1/MinimumMinimummodel/met_weight/add_5:z:03model/met_weight/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:g
"model/met_weight/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
 model/met_weight/clip_by_value_1Maximum,model/met_weight/clip_by_value_1/Minimum:z:0+model/met_weight/clip_by_value_1/y:output:0*
T0*
_output_shapes
:�
model/met_weight/mul_5Mulmodel/met_weight/Cast_1:y:0$model/met_weight/clip_by_value_1:z:0*
T0*
_output_shapes
:a
model/met_weight/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C�
model/met_weight/truediv_3RealDivmodel/met_weight/mul_5:z:0%model/met_weight/truediv_3/y:output:0*
T0*
_output_shapes
:]
model/met_weight/mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/met_weight/mul_6Mul!model/met_weight/mul_6/x:output:0model/met_weight/truediv_3:z:0*
T0*
_output_shapes
:�
!model/met_weight/ReadVariableOp_4ReadVariableOp*model_met_weight_readvariableop_3_resource*
_output_shapes
:*
dtype0m
model/met_weight/Neg_3Neg)model/met_weight/ReadVariableOp_4:value:0*
T0*
_output_shapes
:|
model/met_weight/add_6AddV2model/met_weight/Neg_3:y:0model/met_weight/mul_6:z:0*
T0*
_output_shapes
:]
model/met_weight/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
model/met_weight/mul_7Mul!model/met_weight/mul_7/x:output:0model/met_weight/add_6:z:0*
T0*
_output_shapes
:p
model/met_weight/StopGradient_3StopGradientmodel/met_weight/mul_7:z:0*
T0*
_output_shapes
:�
!model/met_weight/ReadVariableOp_5ReadVariableOp*model_met_weight_readvariableop_3_resource*
_output_shapes
:*
dtype0�
model/met_weight/add_7AddV2)model/met_weight/ReadVariableOp_5:value:0(model/met_weight/StopGradient_3:output:0*
T0*
_output_shapes
:�
model/met_weight/BiasAddBiasAdd#model/met_weight/Reshape_2:output:0model/met_weight/add_7:z:0*
T0*+
_output_shapes
:���������d�
3model/met_weight_minus_one/batchnorm/ReadVariableOpReadVariableOp<model_met_weight_minus_one_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0o
*model/met_weight_minus_one/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(model/met_weight_minus_one/batchnorm/addAddV2;model/met_weight_minus_one/batchnorm/ReadVariableOp:value:03model/met_weight_minus_one/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
*model/met_weight_minus_one/batchnorm/RsqrtRsqrt,model/met_weight_minus_one/batchnorm/add:z:0*
T0*
_output_shapes
:�
7model/met_weight_minus_one/batchnorm/mul/ReadVariableOpReadVariableOp@model_met_weight_minus_one_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
(model/met_weight_minus_one/batchnorm/mulMul.model/met_weight_minus_one/batchnorm/Rsqrt:y:0?model/met_weight_minus_one/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
*model/met_weight_minus_one/batchnorm/mul_1Mul!model/met_weight/BiasAdd:output:0,model/met_weight_minus_one/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������d�
5model/met_weight_minus_one/batchnorm/ReadVariableOp_1ReadVariableOp>model_met_weight_minus_one_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
*model/met_weight_minus_one/batchnorm/mul_2Mul=model/met_weight_minus_one/batchnorm/ReadVariableOp_1:value:0,model/met_weight_minus_one/batchnorm/mul:z:0*
T0*
_output_shapes
:�
5model/met_weight_minus_one/batchnorm/ReadVariableOp_2ReadVariableOp>model_met_weight_minus_one_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
(model/met_weight_minus_one/batchnorm/subSub=model/met_weight_minus_one/batchnorm/ReadVariableOp_2:value:0.model/met_weight_minus_one/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
*model/met_weight_minus_one/batchnorm/add_1AddV2.model/met_weight_minus_one/batchnorm/mul_1:z:0,model/met_weight_minus_one/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������d�
model/multiply/mulMul.model/met_weight_minus_one/batchnorm/add_1:z:0
input_pxpy*
T0*+
_output_shapes
:���������de
#model/output/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
model/output/MeanMeanmodel/multiply/mul:z:0,model/output/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������i
IdentityIdentitymodel/output/Mean:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp3^model/batch_normalization/batchnorm/ReadVariableOp5^model/batch_normalization/batchnorm/ReadVariableOp_15^model/batch_normalization/batchnorm/ReadVariableOp_27^model/batch_normalization/batchnorm/mul/ReadVariableOp5^model/batch_normalization_1/batchnorm/ReadVariableOp7^model/batch_normalization_1/batchnorm/ReadVariableOp_17^model/batch_normalization_1/batchnorm/ReadVariableOp_29^model/batch_normalization_1/batchnorm/mul/ReadVariableOp"^model/embedding0/embedding_lookup"^model/embedding1/embedding_lookup ^model/met_weight/ReadVariableOp"^model/met_weight/ReadVariableOp_1"^model/met_weight/ReadVariableOp_2"^model/met_weight/ReadVariableOp_3"^model/met_weight/ReadVariableOp_4"^model/met_weight/ReadVariableOp_54^model/met_weight_minus_one/batchnorm/ReadVariableOp6^model/met_weight_minus_one/batchnorm/ReadVariableOp_16^model/met_weight_minus_one/batchnorm/ReadVariableOp_28^model/met_weight_minus_one/batchnorm/mul/ReadVariableOp^model/q_dense/ReadVariableOp^model/q_dense/ReadVariableOp_1^model/q_dense/ReadVariableOp_2^model/q_dense/ReadVariableOp_3^model/q_dense/ReadVariableOp_4^model/q_dense/ReadVariableOp_5^model/q_dense_1/ReadVariableOp!^model/q_dense_1/ReadVariableOp_1!^model/q_dense_1/ReadVariableOp_2!^model/q_dense_1/ReadVariableOp_3!^model/q_dense_1/ReadVariableOp_4!^model/q_dense_1/ReadVariableOp_5*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|:���������d:���������d:���������d:���������d: : : : : : : : : : : : : : : : : : : : 2h
2model/batch_normalization/batchnorm/ReadVariableOp2model/batch_normalization/batchnorm/ReadVariableOp2l
4model/batch_normalization/batchnorm/ReadVariableOp_14model/batch_normalization/batchnorm/ReadVariableOp_12l
4model/batch_normalization/batchnorm/ReadVariableOp_24model/batch_normalization/batchnorm/ReadVariableOp_22p
6model/batch_normalization/batchnorm/mul/ReadVariableOp6model/batch_normalization/batchnorm/mul/ReadVariableOp2l
4model/batch_normalization_1/batchnorm/ReadVariableOp4model/batch_normalization_1/batchnorm/ReadVariableOp2p
6model/batch_normalization_1/batchnorm/ReadVariableOp_16model/batch_normalization_1/batchnorm/ReadVariableOp_12p
6model/batch_normalization_1/batchnorm/ReadVariableOp_26model/batch_normalization_1/batchnorm/ReadVariableOp_22t
8model/batch_normalization_1/batchnorm/mul/ReadVariableOp8model/batch_normalization_1/batchnorm/mul/ReadVariableOp2F
!model/embedding0/embedding_lookup!model/embedding0/embedding_lookup2F
!model/embedding1/embedding_lookup!model/embedding1/embedding_lookup2B
model/met_weight/ReadVariableOpmodel/met_weight/ReadVariableOp2F
!model/met_weight/ReadVariableOp_1!model/met_weight/ReadVariableOp_12F
!model/met_weight/ReadVariableOp_2!model/met_weight/ReadVariableOp_22F
!model/met_weight/ReadVariableOp_3!model/met_weight/ReadVariableOp_32F
!model/met_weight/ReadVariableOp_4!model/met_weight/ReadVariableOp_42F
!model/met_weight/ReadVariableOp_5!model/met_weight/ReadVariableOp_52j
3model/met_weight_minus_one/batchnorm/ReadVariableOp3model/met_weight_minus_one/batchnorm/ReadVariableOp2n
5model/met_weight_minus_one/batchnorm/ReadVariableOp_15model/met_weight_minus_one/batchnorm/ReadVariableOp_12n
5model/met_weight_minus_one/batchnorm/ReadVariableOp_25model/met_weight_minus_one/batchnorm/ReadVariableOp_22r
7model/met_weight_minus_one/batchnorm/mul/ReadVariableOp7model/met_weight_minus_one/batchnorm/mul/ReadVariableOp2<
model/q_dense/ReadVariableOpmodel/q_dense/ReadVariableOp2@
model/q_dense/ReadVariableOp_1model/q_dense/ReadVariableOp_12@
model/q_dense/ReadVariableOp_2model/q_dense/ReadVariableOp_22@
model/q_dense/ReadVariableOp_3model/q_dense/ReadVariableOp_32@
model/q_dense/ReadVariableOp_4model/q_dense/ReadVariableOp_42@
model/q_dense/ReadVariableOp_5model/q_dense/ReadVariableOp_52@
model/q_dense_1/ReadVariableOpmodel/q_dense_1/ReadVariableOp2D
 model/q_dense_1/ReadVariableOp_1 model/q_dense_1/ReadVariableOp_12D
 model/q_dense_1/ReadVariableOp_2 model/q_dense_1/ReadVariableOp_22D
 model/q_dense_1/ReadVariableOp_3 model/q_dense_1/ReadVariableOp_32D
 model/q_dense_1/ReadVariableOp_4 model/q_dense_1/ReadVariableOp_42D
 model/q_dense_1/ReadVariableOp_5 model/q_dense_1/ReadVariableOp_5:W S
+
_output_shapes
:���������d
$
_user_specified_name
input_cont:WS
+
_output_shapes
:���������d
$
_user_specified_name
input_pxpy:SO
'
_output_shapes
:���������d
$
_user_specified_name
input_cat0:SO
'
_output_shapes
:���������d
$
_user_specified_name
input_cat1
�
o
E__inference_multiply_layer_call_and_return_conditional_losses_4871154

inputs
inputs_1
identityR
mulMulinputsinputs_1*
T0*+
_output_shapes
:���������dS
IdentityIdentitymul:z:0*
T0*+
_output_shapes
:���������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������d:���������d:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs:SO
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
t
J__inference_concatenate_1_layer_call_and_return_conditional_losses_4870744

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :y
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*+
_output_shapes
:���������d[
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:���������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������d:���������d:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs:SO
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
Q__inference_met_weight_minus_one_layer_call_and_return_conditional_losses_4870630

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
'__inference_model_layer_call_fn_4871201

input_cont

input_pxpy

input_cat0

input_cat1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:$
	unknown_8:$
	unknown_9:$

unknown_10:$

unknown_11:$

unknown_12:$

unknown_13:$

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall
input_cont
input_pxpy
input_cat0
input_cat1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_4871158o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|:���������d:���������d:���������d:���������d: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
+
_output_shapes
:���������d
$
_user_specified_name
input_cont:WS
+
_output_shapes
:���������d
$
_user_specified_name
input_pxpy:SO
'
_output_shapes
:���������d
$
_user_specified_name
input_cat0:SO
'
_output_shapes
:���������d
$
_user_specified_name
input_cat1
�
�
6__inference_met_weight_minus_one_layer_call_fn_4873224

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_met_weight_minus_one_layer_call_and_return_conditional_losses_4870630|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�%
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_4872835

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *��L=�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *��L=�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�	
�
G__inference_embedding0_layer_call_and_return_conditional_losses_4870710

inputs*
embedding_lookup_4870704:
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������d�
embedding_lookupResourceGatherembedding_lookup_4870704Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/4870704*+
_output_shapes
:���������d*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/4870704*+
_output_shapes
:���������d�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������dw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:���������dY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������d: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
��
�
B__inference_model_layer_call_and_return_conditional_losses_4872116
inputs_0
inputs_1
inputs_2
inputs_35
#embedding0_embedding_lookup_4871716:5
#embedding1_embedding_lookup_4871722:1
q_dense_readvariableop_resource:/
!q_dense_readvariableop_3_resource:C
5batch_normalization_batchnorm_readvariableop_resource:G
9batch_normalization_batchnorm_mul_readvariableop_resource:E
7batch_normalization_batchnorm_readvariableop_1_resource:E
7batch_normalization_batchnorm_readvariableop_2_resource:3
!q_dense_1_readvariableop_resource:$1
#q_dense_1_readvariableop_3_resource:$E
7batch_normalization_1_batchnorm_readvariableop_resource:$I
;batch_normalization_1_batchnorm_mul_readvariableop_resource:$G
9batch_normalization_1_batchnorm_readvariableop_1_resource:$G
9batch_normalization_1_batchnorm_readvariableop_2_resource:$4
"met_weight_readvariableop_resource:$2
$met_weight_readvariableop_3_resource:D
6met_weight_minus_one_batchnorm_readvariableop_resource:H
:met_weight_minus_one_batchnorm_mul_readvariableop_resource:F
8met_weight_minus_one_batchnorm_readvariableop_1_resource:F
8met_weight_minus_one_batchnorm_readvariableop_2_resource:
identity��,batch_normalization/batchnorm/ReadVariableOp�.batch_normalization/batchnorm/ReadVariableOp_1�.batch_normalization/batchnorm/ReadVariableOp_2�0batch_normalization/batchnorm/mul/ReadVariableOp�.batch_normalization_1/batchnorm/ReadVariableOp�0batch_normalization_1/batchnorm/ReadVariableOp_1�0batch_normalization_1/batchnorm/ReadVariableOp_2�2batch_normalization_1/batchnorm/mul/ReadVariableOp�embedding0/embedding_lookup�embedding1/embedding_lookup�met_weight/ReadVariableOp�met_weight/ReadVariableOp_1�met_weight/ReadVariableOp_2�met_weight/ReadVariableOp_3�met_weight/ReadVariableOp_4�met_weight/ReadVariableOp_5�-met_weight_minus_one/batchnorm/ReadVariableOp�/met_weight_minus_one/batchnorm/ReadVariableOp_1�/met_weight_minus_one/batchnorm/ReadVariableOp_2�1met_weight_minus_one/batchnorm/mul/ReadVariableOp�q_dense/ReadVariableOp�q_dense/ReadVariableOp_1�q_dense/ReadVariableOp_2�q_dense/ReadVariableOp_3�q_dense/ReadVariableOp_4�q_dense/ReadVariableOp_5�q_dense_1/ReadVariableOp�q_dense_1/ReadVariableOp_1�q_dense_1/ReadVariableOp_2�q_dense_1/ReadVariableOp_3�q_dense_1/ReadVariableOp_4�q_dense_1/ReadVariableOp_5b
embedding0/CastCastinputs_2*

DstT0*

SrcT0*'
_output_shapes
:���������d�
embedding0/embedding_lookupResourceGather#embedding0_embedding_lookup_4871716embedding0/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding0/embedding_lookup/4871716*+
_output_shapes
:���������d*
dtype0�
$embedding0/embedding_lookup/IdentityIdentity$embedding0/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding0/embedding_lookup/4871716*+
_output_shapes
:���������d�
&embedding0/embedding_lookup/Identity_1Identity-embedding0/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������db
embedding1/CastCastinputs_3*

DstT0*

SrcT0*'
_output_shapes
:���������d�
embedding1/embedding_lookupResourceGather#embedding1_embedding_lookup_4871722embedding1/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding1/embedding_lookup/4871722*+
_output_shapes
:���������d*
dtype0�
$embedding1/embedding_lookup/IdentityIdentity$embedding1/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding1/embedding_lookup/4871722*+
_output_shapes
:���������d�
&embedding1/embedding_lookup/Identity_1Identity-embedding1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������dY
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate/concatConcatV2/embedding0/embedding_lookup/Identity_1:output:0/embedding1/embedding_lookup/Identity_1:output:0 concatenate/concat/axis:output:0*
N*
T0*+
_output_shapes
:���������d[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate_1/concatConcatV2inputs_0concatenate/concat:output:0"concatenate_1/concat/axis:output:0*
N*
T0*+
_output_shapes
:���������dO
q_dense/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :O
q_dense/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :c
q_dense/PowPowq_dense/Pow/x:output:0q_dense/Pow/y:output:0*
T0*
_output_shapes
: U
q_dense/CastCastq_dense/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: v
q_dense/ReadVariableOpReadVariableOpq_dense_readvariableop_resource*
_output_shapes

:*
dtype0R
q_dense/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Cs
q_dense/mulMulq_dense/ReadVariableOp:value:0q_dense/mul/y:output:0*
T0*
_output_shapes

:f
q_dense/truedivRealDivq_dense/mul:z:0q_dense/Cast:y:0*
T0*
_output_shapes

:P
q_dense/NegNegq_dense/truediv:z:0*
T0*
_output_shapes

:T
q_dense/RoundRoundq_dense/truediv:z:0*
T0*
_output_shapes

:a
q_dense/addAddV2q_dense/Neg:y:0q_dense/Round:y:0*
T0*
_output_shapes

:^
q_dense/StopGradientStopGradientq_dense/add:z:0*
T0*
_output_shapes

:s
q_dense/add_1AddV2q_dense/truediv:z:0q_dense/StopGradient:output:0*
T0*
_output_shapes

:d
q_dense/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B�
q_dense/clip_by_value/MinimumMinimumq_dense/add_1:z:0(q_dense/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:\
q_dense/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
q_dense/clip_by_valueMaximum!q_dense/clip_by_value/Minimum:z:0 q_dense/clip_by_value/y:output:0*
T0*
_output_shapes

:j
q_dense/mul_1Mulq_dense/Cast:y:0q_dense/clip_by_value:z:0*
T0*
_output_shapes

:X
q_dense/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Cv
q_dense/truediv_1RealDivq_dense/mul_1:z:0q_dense/truediv_1/y:output:0*
T0*
_output_shapes

:T
q_dense/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?n
q_dense/mul_2Mulq_dense/mul_2/x:output:0q_dense/truediv_1:z:0*
T0*
_output_shapes

:x
q_dense/ReadVariableOp_1ReadVariableOpq_dense_readvariableop_resource*
_output_shapes

:*
dtype0_
q_dense/Neg_1Neg q_dense/ReadVariableOp_1:value:0*
T0*
_output_shapes

:e
q_dense/add_2AddV2q_dense/Neg_1:y:0q_dense/mul_2:z:0*
T0*
_output_shapes

:T
q_dense/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?j
q_dense/mul_3Mulq_dense/mul_3/x:output:0q_dense/add_2:z:0*
T0*
_output_shapes

:b
q_dense/StopGradient_1StopGradientq_dense/mul_3:z:0*
T0*
_output_shapes

:x
q_dense/ReadVariableOp_2ReadVariableOpq_dense_readvariableop_resource*
_output_shapes

:*
dtype0�
q_dense/add_3AddV2 q_dense/ReadVariableOp_2:value:0q_dense/StopGradient_1:output:0*
T0*
_output_shapes

:Z
q_dense/ShapeShapeconcatenate_1/concat:output:0*
T0*
_output_shapes
:a
q_dense/unstackUnpackq_dense/Shape:output:0*
T0*
_output_shapes
: : : *	
num`
q_dense/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"      c
q_dense/unstack_1Unpackq_dense/Shape_1:output:0*
T0*
_output_shapes
: : *	
numf
q_dense/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
q_dense/ReshapeReshapeconcatenate_1/concat:output:0q_dense/Reshape/shape:output:0*
T0*'
_output_shapes
:���������g
q_dense/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       {
q_dense/transpose	Transposeq_dense/add_3:z:0q_dense/transpose/perm:output:0*
T0*
_output_shapes

:h
q_dense/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ����~
q_dense/Reshape_1Reshapeq_dense/transpose:y:0 q_dense/Reshape_1/shape:output:0*
T0*
_output_shapes

:�
q_dense/MatMulMatMulq_dense/Reshape:output:0q_dense/Reshape_1:output:0*
T0*'
_output_shapes
:���������[
q_dense/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d[
q_dense/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
q_dense/Reshape_2/shapePackq_dense/unstack:output:0"q_dense/Reshape_2/shape/1:output:0"q_dense/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:�
q_dense/Reshape_2Reshapeq_dense/MatMul:product:0 q_dense/Reshape_2/shape:output:0*
T0*+
_output_shapes
:���������dQ
q_dense/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :Q
q_dense/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :i
q_dense/Pow_1Powq_dense/Pow_1/x:output:0q_dense/Pow_1/y:output:0*
T0*
_output_shapes
: Y
q_dense/Cast_1Castq_dense/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: v
q_dense/ReadVariableOp_3ReadVariableOp!q_dense_readvariableop_3_resource*
_output_shapes
:*
dtype0T
q_dense/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Cu
q_dense/mul_4Mul q_dense/ReadVariableOp_3:value:0q_dense/mul_4/y:output:0*
T0*
_output_shapes
:h
q_dense/truediv_2RealDivq_dense/mul_4:z:0q_dense/Cast_1:y:0*
T0*
_output_shapes
:P
q_dense/Neg_2Negq_dense/truediv_2:z:0*
T0*
_output_shapes
:T
q_dense/Round_1Roundq_dense/truediv_2:z:0*
T0*
_output_shapes
:c
q_dense/add_4AddV2q_dense/Neg_2:y:0q_dense/Round_1:y:0*
T0*
_output_shapes
:^
q_dense/StopGradient_2StopGradientq_dense/add_4:z:0*
T0*
_output_shapes
:s
q_dense/add_5AddV2q_dense/truediv_2:z:0q_dense/StopGradient_2:output:0*
T0*
_output_shapes
:f
!q_dense/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B�
q_dense/clip_by_value_1/MinimumMinimumq_dense/add_5:z:0*q_dense/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:^
q_dense/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
q_dense/clip_by_value_1Maximum#q_dense/clip_by_value_1/Minimum:z:0"q_dense/clip_by_value_1/y:output:0*
T0*
_output_shapes
:j
q_dense/mul_5Mulq_dense/Cast_1:y:0q_dense/clip_by_value_1:z:0*
T0*
_output_shapes
:X
q_dense/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Cr
q_dense/truediv_3RealDivq_dense/mul_5:z:0q_dense/truediv_3/y:output:0*
T0*
_output_shapes
:T
q_dense/mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?j
q_dense/mul_6Mulq_dense/mul_6/x:output:0q_dense/truediv_3:z:0*
T0*
_output_shapes
:v
q_dense/ReadVariableOp_4ReadVariableOp!q_dense_readvariableop_3_resource*
_output_shapes
:*
dtype0[
q_dense/Neg_3Neg q_dense/ReadVariableOp_4:value:0*
T0*
_output_shapes
:a
q_dense/add_6AddV2q_dense/Neg_3:y:0q_dense/mul_6:z:0*
T0*
_output_shapes
:T
q_dense/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?f
q_dense/mul_7Mulq_dense/mul_7/x:output:0q_dense/add_6:z:0*
T0*
_output_shapes
:^
q_dense/StopGradient_3StopGradientq_dense/mul_7:z:0*
T0*
_output_shapes
:v
q_dense/ReadVariableOp_5ReadVariableOp!q_dense_readvariableop_3_resource*
_output_shapes
:*
dtype0~
q_dense/add_7AddV2 q_dense/ReadVariableOp_5:value:0q_dense/StopGradient_3:output:0*
T0*
_output_shapes
:
q_dense/BiasAddBiasAddq_dense/Reshape_2:output:0q_dense/add_7:z:0*
T0*+
_output_shapes
:���������d�
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:x
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:�
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
#batch_normalization/batchnorm/mul_1Mulq_dense/BiasAdd:output:0%batch_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������d�
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:�
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������dT
q_activation/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :T
q_activation/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :r
q_activation/PowPowq_activation/Pow/x:output:0q_activation/Pow/y:output:0*
T0*
_output_shapes
: _
q_activation/CastCastq_activation/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: V
q_activation/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :V
q_activation/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :x
q_activation/Pow_1Powq_activation/Pow_1/x:output:0q_activation/Pow_1/y:output:0*
T0*
_output_shapes
: c
q_activation/Cast_1Castq_activation/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: W
q_activation/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @W
q_activation/Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B :k
q_activation/Cast_2Castq_activation/Cast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: W
q_activation/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   An
q_activation/subSubq_activation/Cast_2:y:0q_activation/sub/y:output:0*
T0*
_output_shapes
: m
q_activation/Pow_2Powq_activation/Const:output:0q_activation/sub:z:0*
T0*
_output_shapes
: k
q_activation/sub_1Subq_activation/Cast_1:y:0q_activation/Pow_2:z:0*
T0*
_output_shapes
: �
q_activation/LessEqual	LessEqual'batch_normalization/batchnorm/add_1:z:0q_activation/sub_1:z:0*
T0*+
_output_shapes
:���������dx
q_activation/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:���������ds
q_activation/ones_like/ShapeShape'batch_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:a
q_activation/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
q_activation/ones_likeFill%q_activation/ones_like/Shape:output:0%q_activation/ones_like/Const:output:0*
T0*+
_output_shapes
:���������dk
q_activation/sub_2Subq_activation/Cast_1:y:0q_activation/Pow_2:z:0*
T0*
_output_shapes
: �
q_activation/mulMulq_activation/ones_like:output:0q_activation/sub_2:z:0*
T0*+
_output_shapes
:���������d�
q_activation/SelectV2SelectV2q_activation/LessEqual:z:0q_activation/Relu:activations:0q_activation/mul:z:0*
T0*+
_output_shapes
:���������d�
q_activation/mul_1Mul'batch_normalization/batchnorm/add_1:z:0q_activation/Cast:y:0*
T0*+
_output_shapes
:���������d�
q_activation/truedivRealDivq_activation/mul_1:z:0q_activation/Cast_1:y:0*
T0*+
_output_shapes
:���������dg
q_activation/NegNegq_activation/truediv:z:0*
T0*+
_output_shapes
:���������dk
q_activation/RoundRoundq_activation/truediv:z:0*
T0*+
_output_shapes
:���������d}
q_activation/addAddV2q_activation/Neg:y:0q_activation/Round:y:0*
T0*+
_output_shapes
:���������du
q_activation/StopGradientStopGradientq_activation/add:z:0*
T0*+
_output_shapes
:���������d�
q_activation/add_1AddV2q_activation/truediv:z:0"q_activation/StopGradient:output:0*
T0*+
_output_shapes
:���������d�
q_activation/truediv_1RealDivq_activation/add_1:z:0q_activation/Cast:y:0*
T0*+
_output_shapes
:���������d]
q_activation/truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?|
q_activation/truediv_2RealDiv!q_activation/truediv_2/x:output:0q_activation/Cast:y:0*
T0*
_output_shapes
: Y
q_activation/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?u
q_activation/sub_3Subq_activation/sub_3/x:output:0q_activation/truediv_2:z:0*
T0*
_output_shapes
: �
"q_activation/clip_by_value/MinimumMinimumq_activation/truediv_1:z:0q_activation/sub_3:z:0*
T0*+
_output_shapes
:���������da
q_activation/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
q_activation/clip_by_valueMaximum&q_activation/clip_by_value/Minimum:z:0%q_activation/clip_by_value/y:output:0*
T0*+
_output_shapes
:���������d�
q_activation/mul_2Mulq_activation/Cast_1:y:0q_activation/clip_by_value:z:0*
T0*+
_output_shapes
:���������do
q_activation/Neg_1Negq_activation/SelectV2:output:0*
T0*+
_output_shapes
:���������d�
q_activation/add_2AddV2q_activation/Neg_1:y:0q_activation/mul_2:z:0*
T0*+
_output_shapes
:���������dY
q_activation/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
q_activation/mul_3Mulq_activation/mul_3/x:output:0q_activation/add_2:z:0*
T0*+
_output_shapes
:���������dy
q_activation/StopGradient_1StopGradientq_activation/mul_3:z:0*
T0*+
_output_shapes
:���������d�
q_activation/add_3AddV2q_activation/SelectV2:output:0$q_activation/StopGradient_1:output:0*
T0*+
_output_shapes
:���������dQ
q_dense_1/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :Q
q_dense_1/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :i
q_dense_1/PowPowq_dense_1/Pow/x:output:0q_dense_1/Pow/y:output:0*
T0*
_output_shapes
: Y
q_dense_1/CastCastq_dense_1/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: z
q_dense_1/ReadVariableOpReadVariableOp!q_dense_1_readvariableop_resource*
_output_shapes

:$*
dtype0T
q_dense_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Cy
q_dense_1/mulMul q_dense_1/ReadVariableOp:value:0q_dense_1/mul/y:output:0*
T0*
_output_shapes

:$l
q_dense_1/truedivRealDivq_dense_1/mul:z:0q_dense_1/Cast:y:0*
T0*
_output_shapes

:$T
q_dense_1/NegNegq_dense_1/truediv:z:0*
T0*
_output_shapes

:$X
q_dense_1/RoundRoundq_dense_1/truediv:z:0*
T0*
_output_shapes

:$g
q_dense_1/addAddV2q_dense_1/Neg:y:0q_dense_1/Round:y:0*
T0*
_output_shapes

:$b
q_dense_1/StopGradientStopGradientq_dense_1/add:z:0*
T0*
_output_shapes

:$y
q_dense_1/add_1AddV2q_dense_1/truediv:z:0q_dense_1/StopGradient:output:0*
T0*
_output_shapes

:$f
!q_dense_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B�
q_dense_1/clip_by_value/MinimumMinimumq_dense_1/add_1:z:0*q_dense_1/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:$^
q_dense_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
q_dense_1/clip_by_valueMaximum#q_dense_1/clip_by_value/Minimum:z:0"q_dense_1/clip_by_value/y:output:0*
T0*
_output_shapes

:$p
q_dense_1/mul_1Mulq_dense_1/Cast:y:0q_dense_1/clip_by_value:z:0*
T0*
_output_shapes

:$Z
q_dense_1/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C|
q_dense_1/truediv_1RealDivq_dense_1/mul_1:z:0q_dense_1/truediv_1/y:output:0*
T0*
_output_shapes

:$V
q_dense_1/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?t
q_dense_1/mul_2Mulq_dense_1/mul_2/x:output:0q_dense_1/truediv_1:z:0*
T0*
_output_shapes

:$|
q_dense_1/ReadVariableOp_1ReadVariableOp!q_dense_1_readvariableop_resource*
_output_shapes

:$*
dtype0c
q_dense_1/Neg_1Neg"q_dense_1/ReadVariableOp_1:value:0*
T0*
_output_shapes

:$k
q_dense_1/add_2AddV2q_dense_1/Neg_1:y:0q_dense_1/mul_2:z:0*
T0*
_output_shapes

:$V
q_dense_1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?p
q_dense_1/mul_3Mulq_dense_1/mul_3/x:output:0q_dense_1/add_2:z:0*
T0*
_output_shapes

:$f
q_dense_1/StopGradient_1StopGradientq_dense_1/mul_3:z:0*
T0*
_output_shapes

:$|
q_dense_1/ReadVariableOp_2ReadVariableOp!q_dense_1_readvariableop_resource*
_output_shapes

:$*
dtype0�
q_dense_1/add_3AddV2"q_dense_1/ReadVariableOp_2:value:0!q_dense_1/StopGradient_1:output:0*
T0*
_output_shapes

:$U
q_dense_1/ShapeShapeq_activation/add_3:z:0*
T0*
_output_shapes
:e
q_dense_1/unstackUnpackq_dense_1/Shape:output:0*
T0*
_output_shapes
: : : *	
numb
q_dense_1/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"   $   g
q_dense_1/unstack_1Unpackq_dense_1/Shape_1:output:0*
T0*
_output_shapes
: : *	
numh
q_dense_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
q_dense_1/ReshapeReshapeq_activation/add_3:z:0 q_dense_1/Reshape/shape:output:0*
T0*'
_output_shapes
:���������i
q_dense_1/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
q_dense_1/transpose	Transposeq_dense_1/add_3:z:0!q_dense_1/transpose/perm:output:0*
T0*
_output_shapes

:$j
q_dense_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   �����
q_dense_1/Reshape_1Reshapeq_dense_1/transpose:y:0"q_dense_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:$�
q_dense_1/MatMulMatMulq_dense_1/Reshape:output:0q_dense_1/Reshape_1:output:0*
T0*'
_output_shapes
:���������$]
q_dense_1/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d]
q_dense_1/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :$�
q_dense_1/Reshape_2/shapePackq_dense_1/unstack:output:0$q_dense_1/Reshape_2/shape/1:output:0$q_dense_1/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:�
q_dense_1/Reshape_2Reshapeq_dense_1/MatMul:product:0"q_dense_1/Reshape_2/shape:output:0*
T0*+
_output_shapes
:���������d$S
q_dense_1/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :S
q_dense_1/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :o
q_dense_1/Pow_1Powq_dense_1/Pow_1/x:output:0q_dense_1/Pow_1/y:output:0*
T0*
_output_shapes
: ]
q_dense_1/Cast_1Castq_dense_1/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: z
q_dense_1/ReadVariableOp_3ReadVariableOp#q_dense_1_readvariableop_3_resource*
_output_shapes
:$*
dtype0V
q_dense_1/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C{
q_dense_1/mul_4Mul"q_dense_1/ReadVariableOp_3:value:0q_dense_1/mul_4/y:output:0*
T0*
_output_shapes
:$n
q_dense_1/truediv_2RealDivq_dense_1/mul_4:z:0q_dense_1/Cast_1:y:0*
T0*
_output_shapes
:$T
q_dense_1/Neg_2Negq_dense_1/truediv_2:z:0*
T0*
_output_shapes
:$X
q_dense_1/Round_1Roundq_dense_1/truediv_2:z:0*
T0*
_output_shapes
:$i
q_dense_1/add_4AddV2q_dense_1/Neg_2:y:0q_dense_1/Round_1:y:0*
T0*
_output_shapes
:$b
q_dense_1/StopGradient_2StopGradientq_dense_1/add_4:z:0*
T0*
_output_shapes
:$y
q_dense_1/add_5AddV2q_dense_1/truediv_2:z:0!q_dense_1/StopGradient_2:output:0*
T0*
_output_shapes
:$h
#q_dense_1/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B�
!q_dense_1/clip_by_value_1/MinimumMinimumq_dense_1/add_5:z:0,q_dense_1/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:$`
q_dense_1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
q_dense_1/clip_by_value_1Maximum%q_dense_1/clip_by_value_1/Minimum:z:0$q_dense_1/clip_by_value_1/y:output:0*
T0*
_output_shapes
:$p
q_dense_1/mul_5Mulq_dense_1/Cast_1:y:0q_dense_1/clip_by_value_1:z:0*
T0*
_output_shapes
:$Z
q_dense_1/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   Cx
q_dense_1/truediv_3RealDivq_dense_1/mul_5:z:0q_dense_1/truediv_3/y:output:0*
T0*
_output_shapes
:$V
q_dense_1/mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?p
q_dense_1/mul_6Mulq_dense_1/mul_6/x:output:0q_dense_1/truediv_3:z:0*
T0*
_output_shapes
:$z
q_dense_1/ReadVariableOp_4ReadVariableOp#q_dense_1_readvariableop_3_resource*
_output_shapes
:$*
dtype0_
q_dense_1/Neg_3Neg"q_dense_1/ReadVariableOp_4:value:0*
T0*
_output_shapes
:$g
q_dense_1/add_6AddV2q_dense_1/Neg_3:y:0q_dense_1/mul_6:z:0*
T0*
_output_shapes
:$V
q_dense_1/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?l
q_dense_1/mul_7Mulq_dense_1/mul_7/x:output:0q_dense_1/add_6:z:0*
T0*
_output_shapes
:$b
q_dense_1/StopGradient_3StopGradientq_dense_1/mul_7:z:0*
T0*
_output_shapes
:$z
q_dense_1/ReadVariableOp_5ReadVariableOp#q_dense_1_readvariableop_3_resource*
_output_shapes
:$*
dtype0�
q_dense_1/add_7AddV2"q_dense_1/ReadVariableOp_5:value:0!q_dense_1/StopGradient_3:output:0*
T0*
_output_shapes
:$�
q_dense_1/BiasAddBiasAddq_dense_1/Reshape_2:output:0q_dense_1/add_7:z:0*
T0*+
_output_shapes
:���������d$�
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:$*
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:$|
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:$�
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:$*
dtype0�
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:$�
%batch_normalization_1/batchnorm/mul_1Mulq_dense_1/BiasAdd:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������d$�
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:$*
dtype0�
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:$�
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:$*
dtype0�
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:$�
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������d$V
q_activation_1/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :V
q_activation_1/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :x
q_activation_1/PowPowq_activation_1/Pow/x:output:0q_activation_1/Pow/y:output:0*
T0*
_output_shapes
: c
q_activation_1/CastCastq_activation_1/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: X
q_activation_1/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :X
q_activation_1/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :~
q_activation_1/Pow_1Powq_activation_1/Pow_1/x:output:0q_activation_1/Pow_1/y:output:0*
T0*
_output_shapes
: g
q_activation_1/Cast_1Castq_activation_1/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: Y
q_activation_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
q_activation_1/Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B :o
q_activation_1/Cast_2Cast q_activation_1/Cast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: Y
q_activation_1/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   At
q_activation_1/subSubq_activation_1/Cast_2:y:0q_activation_1/sub/y:output:0*
T0*
_output_shapes
: s
q_activation_1/Pow_2Powq_activation_1/Const:output:0q_activation_1/sub:z:0*
T0*
_output_shapes
: q
q_activation_1/sub_1Subq_activation_1/Cast_1:y:0q_activation_1/Pow_2:z:0*
T0*
_output_shapes
: �
q_activation_1/LessEqual	LessEqual)batch_normalization_1/batchnorm/add_1:z:0q_activation_1/sub_1:z:0*
T0*+
_output_shapes
:���������d$|
q_activation_1/ReluRelu)batch_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:���������d$w
q_activation_1/ones_like/ShapeShape)batch_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:c
q_activation_1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
q_activation_1/ones_likeFill'q_activation_1/ones_like/Shape:output:0'q_activation_1/ones_like/Const:output:0*
T0*+
_output_shapes
:���������d$q
q_activation_1/sub_2Subq_activation_1/Cast_1:y:0q_activation_1/Pow_2:z:0*
T0*
_output_shapes
: �
q_activation_1/mulMul!q_activation_1/ones_like:output:0q_activation_1/sub_2:z:0*
T0*+
_output_shapes
:���������d$�
q_activation_1/SelectV2SelectV2q_activation_1/LessEqual:z:0!q_activation_1/Relu:activations:0q_activation_1/mul:z:0*
T0*+
_output_shapes
:���������d$�
q_activation_1/mul_1Mul)batch_normalization_1/batchnorm/add_1:z:0q_activation_1/Cast:y:0*
T0*+
_output_shapes
:���������d$�
q_activation_1/truedivRealDivq_activation_1/mul_1:z:0q_activation_1/Cast_1:y:0*
T0*+
_output_shapes
:���������d$k
q_activation_1/NegNegq_activation_1/truediv:z:0*
T0*+
_output_shapes
:���������d$o
q_activation_1/RoundRoundq_activation_1/truediv:z:0*
T0*+
_output_shapes
:���������d$�
q_activation_1/addAddV2q_activation_1/Neg:y:0q_activation_1/Round:y:0*
T0*+
_output_shapes
:���������d$y
q_activation_1/StopGradientStopGradientq_activation_1/add:z:0*
T0*+
_output_shapes
:���������d$�
q_activation_1/add_1AddV2q_activation_1/truediv:z:0$q_activation_1/StopGradient:output:0*
T0*+
_output_shapes
:���������d$�
q_activation_1/truediv_1RealDivq_activation_1/add_1:z:0q_activation_1/Cast:y:0*
T0*+
_output_shapes
:���������d$_
q_activation_1/truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
q_activation_1/truediv_2RealDiv#q_activation_1/truediv_2/x:output:0q_activation_1/Cast:y:0*
T0*
_output_shapes
: [
q_activation_1/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?{
q_activation_1/sub_3Subq_activation_1/sub_3/x:output:0q_activation_1/truediv_2:z:0*
T0*
_output_shapes
: �
$q_activation_1/clip_by_value/MinimumMinimumq_activation_1/truediv_1:z:0q_activation_1/sub_3:z:0*
T0*+
_output_shapes
:���������d$c
q_activation_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
q_activation_1/clip_by_valueMaximum(q_activation_1/clip_by_value/Minimum:z:0'q_activation_1/clip_by_value/y:output:0*
T0*+
_output_shapes
:���������d$�
q_activation_1/mul_2Mulq_activation_1/Cast_1:y:0 q_activation_1/clip_by_value:z:0*
T0*+
_output_shapes
:���������d$s
q_activation_1/Neg_1Neg q_activation_1/SelectV2:output:0*
T0*+
_output_shapes
:���������d$�
q_activation_1/add_2AddV2q_activation_1/Neg_1:y:0q_activation_1/mul_2:z:0*
T0*+
_output_shapes
:���������d$[
q_activation_1/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
q_activation_1/mul_3Mulq_activation_1/mul_3/x:output:0q_activation_1/add_2:z:0*
T0*+
_output_shapes
:���������d$}
q_activation_1/StopGradient_1StopGradientq_activation_1/mul_3:z:0*
T0*+
_output_shapes
:���������d$�
q_activation_1/add_3AddV2 q_activation_1/SelectV2:output:0&q_activation_1/StopGradient_1:output:0*
T0*+
_output_shapes
:���������d$R
met_weight/Pow/xConst*
_output_shapes
: *
dtype0*
value	B :R
met_weight/Pow/yConst*
_output_shapes
: *
dtype0*
value	B :l
met_weight/PowPowmet_weight/Pow/x:output:0met_weight/Pow/y:output:0*
T0*
_output_shapes
: [
met_weight/CastCastmet_weight/Pow:z:0*

DstT0*

SrcT0*
_output_shapes
: |
met_weight/ReadVariableOpReadVariableOp"met_weight_readvariableop_resource*
_output_shapes

:$*
dtype0U
met_weight/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C|
met_weight/mulMul!met_weight/ReadVariableOp:value:0met_weight/mul/y:output:0*
T0*
_output_shapes

:$o
met_weight/truedivRealDivmet_weight/mul:z:0met_weight/Cast:y:0*
T0*
_output_shapes

:$V
met_weight/NegNegmet_weight/truediv:z:0*
T0*
_output_shapes

:$Z
met_weight/RoundRoundmet_weight/truediv:z:0*
T0*
_output_shapes

:$j
met_weight/addAddV2met_weight/Neg:y:0met_weight/Round:y:0*
T0*
_output_shapes

:$d
met_weight/StopGradientStopGradientmet_weight/add:z:0*
T0*
_output_shapes

:$|
met_weight/add_1AddV2met_weight/truediv:z:0 met_weight/StopGradient:output:0*
T0*
_output_shapes

:$g
"met_weight/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B�
 met_weight/clip_by_value/MinimumMinimummet_weight/add_1:z:0+met_weight/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:$_
met_weight/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
met_weight/clip_by_valueMaximum$met_weight/clip_by_value/Minimum:z:0#met_weight/clip_by_value/y:output:0*
T0*
_output_shapes

:$s
met_weight/mul_1Mulmet_weight/Cast:y:0met_weight/clip_by_value:z:0*
T0*
_output_shapes

:$[
met_weight/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C
met_weight/truediv_1RealDivmet_weight/mul_1:z:0met_weight/truediv_1/y:output:0*
T0*
_output_shapes

:$W
met_weight/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?w
met_weight/mul_2Mulmet_weight/mul_2/x:output:0met_weight/truediv_1:z:0*
T0*
_output_shapes

:$~
met_weight/ReadVariableOp_1ReadVariableOp"met_weight_readvariableop_resource*
_output_shapes

:$*
dtype0e
met_weight/Neg_1Neg#met_weight/ReadVariableOp_1:value:0*
T0*
_output_shapes

:$n
met_weight/add_2AddV2met_weight/Neg_1:y:0met_weight/mul_2:z:0*
T0*
_output_shapes

:$W
met_weight/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?s
met_weight/mul_3Mulmet_weight/mul_3/x:output:0met_weight/add_2:z:0*
T0*
_output_shapes

:$h
met_weight/StopGradient_1StopGradientmet_weight/mul_3:z:0*
T0*
_output_shapes

:$~
met_weight/ReadVariableOp_2ReadVariableOp"met_weight_readvariableop_resource*
_output_shapes

:$*
dtype0�
met_weight/add_3AddV2#met_weight/ReadVariableOp_2:value:0"met_weight/StopGradient_1:output:0*
T0*
_output_shapes

:$X
met_weight/ShapeShapeq_activation_1/add_3:z:0*
T0*
_output_shapes
:g
met_weight/unstackUnpackmet_weight/Shape:output:0*
T0*
_output_shapes
: : : *	
numc
met_weight/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"$      i
met_weight/unstack_1Unpackmet_weight/Shape_1:output:0*
T0*
_output_shapes
: : *	
numi
met_weight/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����$   �
met_weight/ReshapeReshapeq_activation_1/add_3:z:0!met_weight/Reshape/shape:output:0*
T0*'
_output_shapes
:���������$j
met_weight/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
met_weight/transpose	Transposemet_weight/add_3:z:0"met_weight/transpose/perm:output:0*
T0*
_output_shapes

:$k
met_weight/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"$   �����
met_weight/Reshape_1Reshapemet_weight/transpose:y:0#met_weight/Reshape_1/shape:output:0*
T0*
_output_shapes

:$�
met_weight/MatMulMatMulmet_weight/Reshape:output:0met_weight/Reshape_1:output:0*
T0*'
_output_shapes
:���������^
met_weight/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d^
met_weight/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
met_weight/Reshape_2/shapePackmet_weight/unstack:output:0%met_weight/Reshape_2/shape/1:output:0%met_weight/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:�
met_weight/Reshape_2Reshapemet_weight/MatMul:product:0#met_weight/Reshape_2/shape:output:0*
T0*+
_output_shapes
:���������dT
met_weight/Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :T
met_weight/Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :r
met_weight/Pow_1Powmet_weight/Pow_1/x:output:0met_weight/Pow_1/y:output:0*
T0*
_output_shapes
: _
met_weight/Cast_1Castmet_weight/Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: |
met_weight/ReadVariableOp_3ReadVariableOp$met_weight_readvariableop_3_resource*
_output_shapes
:*
dtype0W
met_weight/mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C~
met_weight/mul_4Mul#met_weight/ReadVariableOp_3:value:0met_weight/mul_4/y:output:0*
T0*
_output_shapes
:q
met_weight/truediv_2RealDivmet_weight/mul_4:z:0met_weight/Cast_1:y:0*
T0*
_output_shapes
:V
met_weight/Neg_2Negmet_weight/truediv_2:z:0*
T0*
_output_shapes
:Z
met_weight/Round_1Roundmet_weight/truediv_2:z:0*
T0*
_output_shapes
:l
met_weight/add_4AddV2met_weight/Neg_2:y:0met_weight/Round_1:y:0*
T0*
_output_shapes
:d
met_weight/StopGradient_2StopGradientmet_weight/add_4:z:0*
T0*
_output_shapes
:|
met_weight/add_5AddV2met_weight/truediv_2:z:0"met_weight/StopGradient_2:output:0*
T0*
_output_shapes
:i
$met_weight/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �B�
"met_weight/clip_by_value_1/MinimumMinimummet_weight/add_5:z:0-met_weight/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:a
met_weight/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ��
met_weight/clip_by_value_1Maximum&met_weight/clip_by_value_1/Minimum:z:0%met_weight/clip_by_value_1/y:output:0*
T0*
_output_shapes
:s
met_weight/mul_5Mulmet_weight/Cast_1:y:0met_weight/clip_by_value_1:z:0*
T0*
_output_shapes
:[
met_weight/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C{
met_weight/truediv_3RealDivmet_weight/mul_5:z:0met_weight/truediv_3/y:output:0*
T0*
_output_shapes
:W
met_weight/mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?s
met_weight/mul_6Mulmet_weight/mul_6/x:output:0met_weight/truediv_3:z:0*
T0*
_output_shapes
:|
met_weight/ReadVariableOp_4ReadVariableOp$met_weight_readvariableop_3_resource*
_output_shapes
:*
dtype0a
met_weight/Neg_3Neg#met_weight/ReadVariableOp_4:value:0*
T0*
_output_shapes
:j
met_weight/add_6AddV2met_weight/Neg_3:y:0met_weight/mul_6:z:0*
T0*
_output_shapes
:W
met_weight/mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?o
met_weight/mul_7Mulmet_weight/mul_7/x:output:0met_weight/add_6:z:0*
T0*
_output_shapes
:d
met_weight/StopGradient_3StopGradientmet_weight/mul_7:z:0*
T0*
_output_shapes
:|
met_weight/ReadVariableOp_5ReadVariableOp$met_weight_readvariableop_3_resource*
_output_shapes
:*
dtype0�
met_weight/add_7AddV2#met_weight/ReadVariableOp_5:value:0"met_weight/StopGradient_3:output:0*
T0*
_output_shapes
:�
met_weight/BiasAddBiasAddmet_weight/Reshape_2:output:0met_weight/add_7:z:0*
T0*+
_output_shapes
:���������d�
-met_weight_minus_one/batchnorm/ReadVariableOpReadVariableOp6met_weight_minus_one_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0i
$met_weight_minus_one/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
"met_weight_minus_one/batchnorm/addAddV25met_weight_minus_one/batchnorm/ReadVariableOp:value:0-met_weight_minus_one/batchnorm/add/y:output:0*
T0*
_output_shapes
:z
$met_weight_minus_one/batchnorm/RsqrtRsqrt&met_weight_minus_one/batchnorm/add:z:0*
T0*
_output_shapes
:�
1met_weight_minus_one/batchnorm/mul/ReadVariableOpReadVariableOp:met_weight_minus_one_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
"met_weight_minus_one/batchnorm/mulMul(met_weight_minus_one/batchnorm/Rsqrt:y:09met_weight_minus_one/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
$met_weight_minus_one/batchnorm/mul_1Mulmet_weight/BiasAdd:output:0&met_weight_minus_one/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������d�
/met_weight_minus_one/batchnorm/ReadVariableOp_1ReadVariableOp8met_weight_minus_one_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
$met_weight_minus_one/batchnorm/mul_2Mul7met_weight_minus_one/batchnorm/ReadVariableOp_1:value:0&met_weight_minus_one/batchnorm/mul:z:0*
T0*
_output_shapes
:�
/met_weight_minus_one/batchnorm/ReadVariableOp_2ReadVariableOp8met_weight_minus_one_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
"met_weight_minus_one/batchnorm/subSub7met_weight_minus_one/batchnorm/ReadVariableOp_2:value:0(met_weight_minus_one/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
$met_weight_minus_one/batchnorm/add_1AddV2(met_weight_minus_one/batchnorm/mul_1:z:0&met_weight_minus_one/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������d}
multiply/mulMul(met_weight_minus_one/batchnorm/add_1:z:0inputs_1*
T0*+
_output_shapes
:���������d_
output/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
output/MeanMeanmultiply/mul:z:0&output/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������c
IdentityIdentityoutput/Mean:output:0^NoOp*
T0*'
_output_shapes
:����������	
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp^embedding0/embedding_lookup^embedding1/embedding_lookup^met_weight/ReadVariableOp^met_weight/ReadVariableOp_1^met_weight/ReadVariableOp_2^met_weight/ReadVariableOp_3^met_weight/ReadVariableOp_4^met_weight/ReadVariableOp_5.^met_weight_minus_one/batchnorm/ReadVariableOp0^met_weight_minus_one/batchnorm/ReadVariableOp_10^met_weight_minus_one/batchnorm/ReadVariableOp_22^met_weight_minus_one/batchnorm/mul/ReadVariableOp^q_dense/ReadVariableOp^q_dense/ReadVariableOp_1^q_dense/ReadVariableOp_2^q_dense/ReadVariableOp_3^q_dense/ReadVariableOp_4^q_dense/ReadVariableOp_5^q_dense_1/ReadVariableOp^q_dense_1/ReadVariableOp_1^q_dense_1/ReadVariableOp_2^q_dense_1/ReadVariableOp_3^q_dense_1/ReadVariableOp_4^q_dense_1/ReadVariableOp_5*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|:���������d:���������d:���������d:���������d: : : : : : : : : : : : : : : : : : : : 2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2`
.batch_normalization/batchnorm/ReadVariableOp_1.batch_normalization/batchnorm/ReadVariableOp_12`
.batch_normalization/batchnorm/ReadVariableOp_2.batch_normalization/batchnorm/ReadVariableOp_22d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2d
0batch_normalization_1/batchnorm/ReadVariableOp_10batch_normalization_1/batchnorm/ReadVariableOp_12d
0batch_normalization_1/batchnorm/ReadVariableOp_20batch_normalization_1/batchnorm/ReadVariableOp_22h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2:
embedding0/embedding_lookupembedding0/embedding_lookup2:
embedding1/embedding_lookupembedding1/embedding_lookup26
met_weight/ReadVariableOpmet_weight/ReadVariableOp2:
met_weight/ReadVariableOp_1met_weight/ReadVariableOp_12:
met_weight/ReadVariableOp_2met_weight/ReadVariableOp_22:
met_weight/ReadVariableOp_3met_weight/ReadVariableOp_32:
met_weight/ReadVariableOp_4met_weight/ReadVariableOp_42:
met_weight/ReadVariableOp_5met_weight/ReadVariableOp_52^
-met_weight_minus_one/batchnorm/ReadVariableOp-met_weight_minus_one/batchnorm/ReadVariableOp2b
/met_weight_minus_one/batchnorm/ReadVariableOp_1/met_weight_minus_one/batchnorm/ReadVariableOp_12b
/met_weight_minus_one/batchnorm/ReadVariableOp_2/met_weight_minus_one/batchnorm/ReadVariableOp_22f
1met_weight_minus_one/batchnorm/mul/ReadVariableOp1met_weight_minus_one/batchnorm/mul/ReadVariableOp20
q_dense/ReadVariableOpq_dense/ReadVariableOp24
q_dense/ReadVariableOp_1q_dense/ReadVariableOp_124
q_dense/ReadVariableOp_2q_dense/ReadVariableOp_224
q_dense/ReadVariableOp_3q_dense/ReadVariableOp_324
q_dense/ReadVariableOp_4q_dense/ReadVariableOp_424
q_dense/ReadVariableOp_5q_dense/ReadVariableOp_524
q_dense_1/ReadVariableOpq_dense_1/ReadVariableOp28
q_dense_1/ReadVariableOp_1q_dense_1/ReadVariableOp_128
q_dense_1/ReadVariableOp_2q_dense_1/ReadVariableOp_228
q_dense_1/ReadVariableOp_3q_dense_1/ReadVariableOp_328
q_dense_1/ReadVariableOp_4q_dense_1/ReadVariableOp_428
q_dense_1/ReadVariableOp_5q_dense_1/ReadVariableOp_5:U Q
+
_output_shapes
:���������d
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������d
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������d
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:���������d
"
_user_specified_name
inputs/3
�
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_4872801

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�!
g
K__inference_q_activation_1_layer_call_and_return_conditional_losses_4873117

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B :Q
Cast_2CastCast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   AG
subSub
Cast_2:y:0sub/y:output:0*
T0*
_output_shapes
: F
Pow_2PowConst:output:0sub:z:0*
T0*
_output_shapes
: D
sub_1Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: _
	LessEqual	LessEqualinputs	sub_1:z:0*
T0*+
_output_shapes
:���������d$J
ReluReluinputs*
T0*+
_output_shapes
:���������d$E
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?{
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*+
_output_shapes
:���������d$D
sub_2Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: _
mulMulones_like:output:0	sub_2:z:0*
T0*+
_output_shapes
:���������d$v
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*+
_output_shapes
:���������d$T
mul_1MulinputsCast:y:0*
T0*+
_output_shapes
:���������d$_
truedivRealDiv	mul_1:z:0
Cast_1:y:0*
T0*+
_output_shapes
:���������d$M
NegNegtruediv:z:0*
T0*+
_output_shapes
:���������d$Q
RoundRoundtruediv:z:0*
T0*+
_output_shapes
:���������d$V
addAddV2Neg:y:0	Round:y:0*
T0*+
_output_shapes
:���������d$[
StopGradientStopGradientadd:z:0*
T0*+
_output_shapes
:���������d$h
add_1AddV2truediv:z:0StopGradient:output:0*
T0*+
_output_shapes
:���������d$_
	truediv_1RealDiv	add_1:z:0Cast:y:0*
T0*+
_output_shapes
:���������d$P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
	truediv_2RealDivtruediv_2/x:output:0Cast:y:0*
T0*
_output_shapes
: L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_3Subsub_3/x:output:0truediv_2:z:0*
T0*
_output_shapes
: p
clip_by_value/MinimumMinimumtruediv_1:z:0	sub_3:z:0*
T0*+
_output_shapes
:���������d$T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*+
_output_shapes
:���������d$a
mul_2Mul
Cast_1:y:0clip_by_value:z:0*
T0*+
_output_shapes
:���������d$U
Neg_1NegSelectV2:output:0*
T0*+
_output_shapes
:���������d$Z
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*+
_output_shapes
:���������d$L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?_
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*+
_output_shapes
:���������d$_
StopGradient_1StopGradient	mul_3:z:0*
T0*+
_output_shapes
:���������d$p
add_3AddV2SelectV2:output:0StopGradient_1:output:0*
T0*+
_output_shapes
:���������d$U
IdentityIdentity	add_3:z:0*
T0*+
_output_shapes
:���������d$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d$:S O
+
_output_shapes
:���������d$
 
_user_specified_nameinputs
�	
�
G__inference_embedding0_layer_call_and_return_conditional_losses_4872618

inputs*
embedding_lookup_4872612:
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������d�
embedding_lookupResourceGatherembedding_lookup_4872612Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/4872612*+
_output_shapes
:���������d*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/4872612*+
_output_shapes
:���������d�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������dw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:���������dY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������d: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�<
�
G__inference_met_weight_layer_call_and_return_conditional_losses_4873211

inputs)
readvariableop_resource:$'
readvariableop_3_resource:
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:$*
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C[
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes

:$N
truedivRealDivmul:z:0Cast:y:0*
T0*
_output_shapes

:$@
NegNegtruediv:z:0*
T0*
_output_shapes

:$D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:$I
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:$N
StopGradientStopGradientadd:z:0*
T0*
_output_shapes

:$[
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:$\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Bv
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:$T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:$R
mul_1MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:$P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C^
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*
_output_shapes

:$L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*
_output_shapes

:$h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:$*
dtype0O
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes

:$M
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*
_output_shapes

:$L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*
_output_shapes

:$R
StopGradient_1StopGradient	mul_3:z:0*
T0*
_output_shapes

:$h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:$*
dtype0j
add_3AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*
_output_shapes

:$;
ShapeShapeinputs*
T0*
_output_shapes
:Q
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
numX
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"$      S
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����$   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:���������$_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       c
	transpose	Transpose	add_3:z:0transpose/perm:output:0*
T0*
_output_shapes

:$`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"$   ����f
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:$h
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:���������S
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :dS
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:v
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:���������dI
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C]
mul_4MulReadVariableOp_3:value:0mul_4/y:output:0*
T0*
_output_shapes
:P
	truediv_2RealDiv	mul_4:z:0
Cast_1:y:0*
T0*
_output_shapes
:@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:K
add_4AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:N
StopGradient_2StopGradient	add_4:z:0*
T0*
_output_shapes
:[
add_5AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Bv
clip_by_value_1/MinimumMinimum	add_5:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:R
mul_5Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   CZ
	truediv_3RealDiv	mul_5:z:0truediv_3/y:output:0*
T0*
_output_shapes
:L
mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_6Mulmul_6/x:output:0truediv_3:z:0*
T0*
_output_shapes
:f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0K
Neg_3NegReadVariableOp_4:value:0*
T0*
_output_shapes
:I
add_6AddV2	Neg_3:y:0	mul_6:z:0*
T0*
_output_shapes
:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_7Mulmul_7/x:output:0	add_6:z:0*
T0*
_output_shapes
:N
StopGradient_3StopGradient	mul_7:z:0*
T0*
_output_shapes
:f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype0f
add_7AddV2ReadVariableOp_5:value:0StopGradient_3:output:0*
T0*
_output_shapes
:g
BiasAddBiasAddReshape_2:output:0	add_7:z:0*
T0*+
_output_shapes
:���������dc
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:���������d�
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������d$: : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_5:S O
+
_output_shapes
:���������d$
 
_user_specified_nameinputs
�
�
6__inference_met_weight_minus_one_layer_call_fn_4873237

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_met_weight_minus_one_layer_call_and_return_conditional_losses_4870663|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
q
E__inference_multiply_layer_call_and_return_conditional_losses_4873289
inputs_0
inputs_1
identityT
mulMulinputs_0inputs_1*
T0*+
_output_shapes
:���������dS
IdentityIdentitymul:z:0*
T0*+
_output_shapes
:���������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������d:���������d:U Q
+
_output_shapes
:���������d
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������d
"
_user_specified_name
inputs/1
�
�
'__inference_model_layer_call_fn_4871709
inputs_0
inputs_1
inputs_2
inputs_3
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:$
	unknown_8:$
	unknown_9:$

unknown_10:$

unknown_11:$

unknown_12:$

unknown_13:$

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_4871394o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|:���������d:���������d:���������d:���������d: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:���������d
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������d
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:���������d
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:���������d
"
_user_specified_name
inputs/3
�
r
H__inference_concatenate_layer_call_and_return_conditional_losses_4870735

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :y
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*+
_output_shapes
:���������d[
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:���������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������d:���������d:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs:SO
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_4870466

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
[
/__inference_concatenate_1_layer_call_fn_4872654
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_4870744d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������d:���������d:U Q
+
_output_shapes
:���������d
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������d
"
_user_specified_name
inputs/1
�
t
H__inference_concatenate_layer_call_and_return_conditional_losses_4872648
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :{
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*+
_output_shapes
:���������d[
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:���������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������d:���������d:U Q
+
_output_shapes
:���������d
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������d
"
_user_specified_name
inputs/1
�
�
Q__inference_met_weight_minus_one_layer_call_and_return_conditional_losses_4873257

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
V
*__inference_multiply_layer_call_fn_4873283
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_multiply_layer_call_and_return_conditional_losses_4871154d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������d:���������d:U Q
+
_output_shapes
:���������d
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������d
"
_user_specified_name
inputs/1
�n
�
 __inference__traced_save_4873491
file_prefix4
0savev2_embedding0_embeddings_read_readvariableop4
0savev2_embedding1_embeddings_read_readvariableop-
)savev2_q_dense_kernel_read_readvariableop+
'savev2_q_dense_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop/
+savev2_q_dense_1_kernel_read_readvariableop-
)savev2_q_dense_1_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop0
,savev2_met_weight_kernel_read_readvariableop.
*savev2_met_weight_bias_read_readvariableop9
5savev2_met_weight_minus_one_gamma_read_readvariableop8
4savev2_met_weight_minus_one_beta_read_readvariableop?
;savev2_met_weight_minus_one_moving_mean_read_readvariableopC
?savev2_met_weight_minus_one_moving_variance_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop;
7savev2_adam_embedding0_embeddings_m_read_readvariableop;
7savev2_adam_embedding1_embeddings_m_read_readvariableop4
0savev2_adam_q_dense_kernel_m_read_readvariableop2
.savev2_adam_q_dense_bias_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableop6
2savev2_adam_q_dense_1_kernel_m_read_readvariableop4
0savev2_adam_q_dense_1_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_m_read_readvariableop7
3savev2_adam_met_weight_kernel_m_read_readvariableop5
1savev2_adam_met_weight_bias_m_read_readvariableop;
7savev2_adam_embedding0_embeddings_v_read_readvariableop;
7savev2_adam_embedding1_embeddings_v_read_readvariableop4
0savev2_adam_q_dense_kernel_v_read_readvariableop2
.savev2_adam_q_dense_bias_v_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableop6
2savev2_adam_q_dense_1_kernel_v_read_readvariableop4
0savev2_adam_q_dense_1_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_v_read_readvariableop7
3savev2_adam_met_weight_kernel_v_read_readvariableop5
1savev2_adam_met_weight_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*�
value�B�8B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*�
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_embedding0_embeddings_read_readvariableop0savev2_embedding1_embeddings_read_readvariableop)savev2_q_dense_kernel_read_readvariableop'savev2_q_dense_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop+savev2_q_dense_1_kernel_read_readvariableop)savev2_q_dense_1_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop,savev2_met_weight_kernel_read_readvariableop*savev2_met_weight_bias_read_readvariableop5savev2_met_weight_minus_one_gamma_read_readvariableop4savev2_met_weight_minus_one_beta_read_readvariableop;savev2_met_weight_minus_one_moving_mean_read_readvariableop?savev2_met_weight_minus_one_moving_variance_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop7savev2_adam_embedding0_embeddings_m_read_readvariableop7savev2_adam_embedding1_embeddings_m_read_readvariableop0savev2_adam_q_dense_kernel_m_read_readvariableop.savev2_adam_q_dense_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop2savev2_adam_q_dense_1_kernel_m_read_readvariableop0savev2_adam_q_dense_1_bias_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop3savev2_adam_met_weight_kernel_m_read_readvariableop1savev2_adam_met_weight_bias_m_read_readvariableop7savev2_adam_embedding0_embeddings_v_read_readvariableop7savev2_adam_embedding1_embeddings_v_read_readvariableop0savev2_adam_q_dense_kernel_v_read_readvariableop.savev2_adam_q_dense_bias_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop2savev2_adam_q_dense_1_kernel_v_read_readvariableop0savev2_adam_q_dense_1_bias_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop3savev2_adam_met_weight_kernel_v_read_readvariableop1savev2_adam_met_weight_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *F
dtypes<
:28	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :::::::::$:$:$:$:$:$:$:::::: : : : : : : : : : : :::::::$:$:$:$:$::::::::$:$:$:$:$:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$	 

_output_shapes

:$: 


_output_shapes
:$: 

_output_shapes
:$: 

_output_shapes
:$: 

_output_shapes
:$: 

_output_shapes
:$:$ 

_output_shapes

:$: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$  

_output_shapes

::$! 

_output_shapes

::$" 

_output_shapes

:: #

_output_shapes
:: $

_output_shapes
:: %

_output_shapes
::$& 

_output_shapes

:$: '

_output_shapes
:$: (

_output_shapes
:$: )

_output_shapes
:$:$* 

_output_shapes

:$: +

_output_shapes
::$, 

_output_shapes

::$- 

_output_shapes

::$. 

_output_shapes

:: /

_output_shapes
:: 0

_output_shapes
:: 1

_output_shapes
::$2 

_output_shapes

:$: 3

_output_shapes
:$: 4

_output_shapes
:$: 5

_output_shapes
:$:$6 

_output_shapes

:$: 7

_output_shapes
::8

_output_shapes
: 
�
�
Q__inference_met_weight_minus_one_layer_call_and_return_conditional_losses_4870663

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
,__inference_embedding0_layer_call_fn_4872608

inputs
unknown:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_embedding0_layer_call_and_return_conditional_losses_4870710s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������d: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
Q__inference_met_weight_minus_one_layer_call_and_return_conditional_losses_4873277

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
J
.__inference_q_activation_layer_call_fn_4872840

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_q_activation_layer_call_and_return_conditional_losses_4870895d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�
_
C__inference_output_layer_call_and_return_conditional_losses_4873300

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:������������������^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�!
g
K__inference_q_activation_1_layer_call_and_return_conditional_losses_4871046

inputs
identityG
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
Cast_2/xConst*
_output_shapes
: *
dtype0*
value	B :Q
Cast_2CastCast_2/x:output:0*

DstT0*

SrcT0*
_output_shapes
: J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *   AG
subSub
Cast_2:y:0sub/y:output:0*
T0*
_output_shapes
: F
Pow_2PowConst:output:0sub:z:0*
T0*
_output_shapes
: D
sub_1Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: _
	LessEqual	LessEqualinputs	sub_1:z:0*
T0*+
_output_shapes
:���������d$J
ReluReluinputs*
T0*+
_output_shapes
:���������d$E
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?{
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*+
_output_shapes
:���������d$D
sub_2Sub
Cast_1:y:0	Pow_2:z:0*
T0*
_output_shapes
: _
mulMulones_like:output:0	sub_2:z:0*
T0*+
_output_shapes
:���������d$v
SelectV2SelectV2LessEqual:z:0Relu:activations:0mul:z:0*
T0*+
_output_shapes
:���������d$T
mul_1MulinputsCast:y:0*
T0*+
_output_shapes
:���������d$_
truedivRealDiv	mul_1:z:0
Cast_1:y:0*
T0*+
_output_shapes
:���������d$M
NegNegtruediv:z:0*
T0*+
_output_shapes
:���������d$Q
RoundRoundtruediv:z:0*
T0*+
_output_shapes
:���������d$V
addAddV2Neg:y:0	Round:y:0*
T0*+
_output_shapes
:���������d$[
StopGradientStopGradientadd:z:0*
T0*+
_output_shapes
:���������d$h
add_1AddV2truediv:z:0StopGradient:output:0*
T0*+
_output_shapes
:���������d$_
	truediv_1RealDiv	add_1:z:0Cast:y:0*
T0*+
_output_shapes
:���������d$P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
	truediv_2RealDivtruediv_2/x:output:0Cast:y:0*
T0*
_output_shapes
: L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
sub_3Subsub_3/x:output:0truediv_2:z:0*
T0*
_output_shapes
: p
clip_by_value/MinimumMinimumtruediv_1:z:0	sub_3:z:0*
T0*+
_output_shapes
:���������d$T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*+
_output_shapes
:���������d$a
mul_2Mul
Cast_1:y:0clip_by_value:z:0*
T0*+
_output_shapes
:���������d$U
Neg_1NegSelectV2:output:0*
T0*+
_output_shapes
:���������d$Z
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*+
_output_shapes
:���������d$L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?_
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*+
_output_shapes
:���������d$_
StopGradient_1StopGradient	mul_3:z:0*
T0*+
_output_shapes
:���������d$p
add_3AddV2SelectV2:output:0StopGradient_1:output:0*
T0*+
_output_shapes
:���������d$U
IdentityIdentity	add_3:z:0*
T0*+
_output_shapes
:���������d$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d$:S O
+
_output_shapes
:���������d$
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_layer_call_fn_4872768

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_4870466|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�	
�
G__inference_embedding1_layer_call_and_return_conditional_losses_4870724

inputs*
embedding_lookup_4870718:
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������d�
embedding_lookupResourceGatherembedding_lookup_4870718Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/4870718*+
_output_shapes
:���������d*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/4870718*+
_output_shapes
:���������d�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������dw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:���������dY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������d: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�D
�	
B__inference_model_layer_call_and_return_conditional_losses_4871394

inputs
inputs_1
inputs_2
inputs_3$
embedding0_4871339:$
embedding1_4871342:!
q_dense_4871347:
q_dense_4871349:)
batch_normalization_4871352:)
batch_normalization_4871354:)
batch_normalization_4871356:)
batch_normalization_4871358:#
q_dense_1_4871362:$
q_dense_1_4871364:$+
batch_normalization_1_4871367:$+
batch_normalization_1_4871369:$+
batch_normalization_1_4871371:$+
batch_normalization_1_4871373:$$
met_weight_4871377:$ 
met_weight_4871379:*
met_weight_minus_one_4871382:*
met_weight_minus_one_4871384:*
met_weight_minus_one_4871386:*
met_weight_minus_one_4871388:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�"embedding0/StatefulPartitionedCall�"embedding1/StatefulPartitionedCall�"met_weight/StatefulPartitionedCall�,met_weight_minus_one/StatefulPartitionedCall�q_dense/StatefulPartitionedCall�!q_dense_1/StatefulPartitionedCall�
"embedding0/StatefulPartitionedCallStatefulPartitionedCallinputs_2embedding0_4871339*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_embedding0_layer_call_and_return_conditional_losses_4870710�
"embedding1/StatefulPartitionedCallStatefulPartitionedCallinputs_3embedding1_4871342*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_embedding1_layer_call_and_return_conditional_losses_4870724�
concatenate/PartitionedCallPartitionedCall+embedding0/StatefulPartitionedCall:output:0+embedding1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_4870735�
concatenate_1/PartitionedCallPartitionedCallinputs$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_4870744�
q_dense/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0q_dense_4871347q_dense_4871349*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_q_dense_layer_call_and_return_conditional_losses_4870831�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall(q_dense/StatefulPartitionedCall:output:0batch_normalization_4871352batch_normalization_4871354batch_normalization_4871356batch_normalization_4871358*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_4870513�
q_activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_q_activation_layer_call_and_return_conditional_losses_4870895�
!q_dense_1/StatefulPartitionedCallStatefulPartitionedCall%q_activation/PartitionedCall:output:0q_dense_1_4871362q_dense_1_4871364*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_q_dense_1_layer_call_and_return_conditional_losses_4870982�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall*q_dense_1/StatefulPartitionedCall:output:0batch_normalization_1_4871367batch_normalization_1_4871369batch_normalization_1_4871371batch_normalization_1_4871373*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4870595�
q_activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_q_activation_1_layer_call_and_return_conditional_losses_4871046�
"met_weight/StatefulPartitionedCallStatefulPartitionedCall'q_activation_1/PartitionedCall:output:0met_weight_4871377met_weight_4871379*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_met_weight_layer_call_and_return_conditional_losses_4871133�
,met_weight_minus_one/StatefulPartitionedCallStatefulPartitionedCall+met_weight/StatefulPartitionedCall:output:0met_weight_minus_one_4871382met_weight_minus_one_4871384met_weight_minus_one_4871386met_weight_minus_one_4871388*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_met_weight_minus_one_layer_call_and_return_conditional_losses_4870663�
multiply/PartitionedCallPartitionedCall5met_weight_minus_one/StatefulPartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_multiply_layer_call_and_return_conditional_losses_4871154�
output/PartitionedCallPartitionedCall!multiply/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_4870684n
IdentityIdentityoutput/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall#^embedding0/StatefulPartitionedCall#^embedding1/StatefulPartitionedCall#^met_weight/StatefulPartitionedCall-^met_weight_minus_one/StatefulPartitionedCall ^q_dense/StatefulPartitionedCall"^q_dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|:���������d:���������d:���������d:���������d: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2H
"embedding0/StatefulPartitionedCall"embedding0/StatefulPartitionedCall2H
"embedding1/StatefulPartitionedCall"embedding1/StatefulPartitionedCall2H
"met_weight/StatefulPartitionedCall"met_weight/StatefulPartitionedCall2\
,met_weight_minus_one/StatefulPartitionedCall,met_weight_minus_one/StatefulPartitionedCall2B
q_dense/StatefulPartitionedCallq_dense/StatefulPartitionedCall2F
!q_dense_1/StatefulPartitionedCall!q_dense_1/StatefulPartitionedCall:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs:SO
+
_output_shapes
:���������d
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������d
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
_
C__inference_output_layer_call_and_return_conditional_losses_4870684

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:������������������^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
v
J__inference_concatenate_1_layer_call_and_return_conditional_losses_4872661
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :{
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*+
_output_shapes
:���������d[
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:���������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������d:���������d:U Q
+
_output_shapes
:���������d
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������d
"
_user_specified_name
inputs/1
�%
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_4870513

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *��L=�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *��L=�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�%
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4873063

inputs5
'assignmovingavg_readvariableop_resource:$7
)assignmovingavg_1_readvariableop_resource:$3
%batchnorm_mul_readvariableop_resource:$/
!batchnorm_readvariableop_resource:$
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:$*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:$�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :������������������$s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:$*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:$*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:$*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *��L=�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:$*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:$x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:$�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *��L=�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:$*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:$~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:$�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:$P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:$~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:$*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:$p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :������������������$h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:$v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:$*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:$
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :������������������$o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :������������������$�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������$: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :������������������$
 
_user_specified_nameinputs
�
�
,__inference_met_weight_layer_call_fn_4873126

inputs
unknown:$
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_met_weight_layer_call_and_return_conditional_losses_4871133s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������d$: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������d$
 
_user_specified_nameinputs
�
Y
-__inference_concatenate_layer_call_fn_4872641
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_4870735d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������d:���������d:U Q
+
_output_shapes
:���������d
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������d
"
_user_specified_name
inputs/1
�
D
(__inference_output_layer_call_fn_4873294

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_4870684i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�	
�
G__inference_embedding1_layer_call_and_return_conditional_losses_4872635

inputs*
embedding_lookup_4872629:
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:���������d�
embedding_lookupResourceGatherembedding_lookup_4872629Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/4872629*+
_output_shapes
:���������d*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/4872629*+
_output_shapes
:���������d�
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������dw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:���������dY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������d: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�E
�

B__inference_model_layer_call_and_return_conditional_losses_4871546

input_cont

input_pxpy

input_cat0

input_cat1$
embedding0_4871491:$
embedding1_4871494:!
q_dense_4871499:
q_dense_4871501:)
batch_normalization_4871504:)
batch_normalization_4871506:)
batch_normalization_4871508:)
batch_normalization_4871510:#
q_dense_1_4871514:$
q_dense_1_4871516:$+
batch_normalization_1_4871519:$+
batch_normalization_1_4871521:$+
batch_normalization_1_4871523:$+
batch_normalization_1_4871525:$$
met_weight_4871529:$ 
met_weight_4871531:*
met_weight_minus_one_4871534:*
met_weight_minus_one_4871536:*
met_weight_minus_one_4871538:*
met_weight_minus_one_4871540:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�"embedding0/StatefulPartitionedCall�"embedding1/StatefulPartitionedCall�"met_weight/StatefulPartitionedCall�,met_weight_minus_one/StatefulPartitionedCall�q_dense/StatefulPartitionedCall�!q_dense_1/StatefulPartitionedCall�
"embedding0/StatefulPartitionedCallStatefulPartitionedCall
input_cat0embedding0_4871491*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_embedding0_layer_call_and_return_conditional_losses_4870710�
"embedding1/StatefulPartitionedCallStatefulPartitionedCall
input_cat1embedding1_4871494*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_embedding1_layer_call_and_return_conditional_losses_4870724�
concatenate/PartitionedCallPartitionedCall+embedding0/StatefulPartitionedCall:output:0+embedding1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_4870735�
concatenate_1/PartitionedCallPartitionedCall
input_cont$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_4870744�
q_dense/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0q_dense_4871499q_dense_4871501*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_q_dense_layer_call_and_return_conditional_losses_4870831�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall(q_dense/StatefulPartitionedCall:output:0batch_normalization_4871504batch_normalization_4871506batch_normalization_4871508batch_normalization_4871510*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_4870466�
q_activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_q_activation_layer_call_and_return_conditional_losses_4870895�
!q_dense_1/StatefulPartitionedCallStatefulPartitionedCall%q_activation/PartitionedCall:output:0q_dense_1_4871514q_dense_1_4871516*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_q_dense_1_layer_call_and_return_conditional_losses_4870982�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall*q_dense_1/StatefulPartitionedCall:output:0batch_normalization_1_4871519batch_normalization_1_4871521batch_normalization_1_4871523batch_normalization_1_4871525*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d$*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4870548�
q_activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_q_activation_1_layer_call_and_return_conditional_losses_4871046�
"met_weight/StatefulPartitionedCallStatefulPartitionedCall'q_activation_1/PartitionedCall:output:0met_weight_4871529met_weight_4871531*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_met_weight_layer_call_and_return_conditional_losses_4871133�
,met_weight_minus_one/StatefulPartitionedCallStatefulPartitionedCall+met_weight/StatefulPartitionedCall:output:0met_weight_minus_one_4871534met_weight_minus_one_4871536met_weight_minus_one_4871538met_weight_minus_one_4871540*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_met_weight_minus_one_layer_call_and_return_conditional_losses_4870630�
multiply/PartitionedCallPartitionedCall5met_weight_minus_one/StatefulPartitionedCall:output:0
input_pxpy*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_multiply_layer_call_and_return_conditional_losses_4871154�
output/PartitionedCallPartitionedCall!multiply/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_4870684n
IdentityIdentityoutput/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall#^embedding0/StatefulPartitionedCall#^embedding1/StatefulPartitionedCall#^met_weight/StatefulPartitionedCall-^met_weight_minus_one/StatefulPartitionedCall ^q_dense/StatefulPartitionedCall"^q_dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|:���������d:���������d:���������d:���������d: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2H
"embedding0/StatefulPartitionedCall"embedding0/StatefulPartitionedCall2H
"embedding1/StatefulPartitionedCall"embedding1/StatefulPartitionedCall2H
"met_weight/StatefulPartitionedCall"met_weight/StatefulPartitionedCall2\
,met_weight_minus_one/StatefulPartitionedCall,met_weight_minus_one/StatefulPartitionedCall2B
q_dense/StatefulPartitionedCallq_dense/StatefulPartitionedCall2F
!q_dense_1/StatefulPartitionedCall!q_dense_1/StatefulPartitionedCall:W S
+
_output_shapes
:���������d
$
_user_specified_name
input_cont:WS
+
_output_shapes
:���������d
$
_user_specified_name
input_pxpy:SO
'
_output_shapes
:���������d
$
_user_specified_name
input_cat0:SO
'
_output_shapes
:���������d
$
_user_specified_name
input_cat1
�<
�
F__inference_q_dense_1_layer_call_and_return_conditional_losses_4870982

inputs)
readvariableop_resource:$'
readvariableop_3_resource:$
identity��ReadVariableOp�ReadVariableOp_1�ReadVariableOp_2�ReadVariableOp_3�ReadVariableOp_4�ReadVariableOp_5G
Pow/xConst*
_output_shapes
: *
dtype0*
value	B :G
Pow/yConst*
_output_shapes
: *
dtype0*
value	B :K
PowPowPow/x:output:0Pow/y:output:0*
T0*
_output_shapes
: E
CastCastPow:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:$*
dtype0J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C[
mulMulReadVariableOp:value:0mul/y:output:0*
T0*
_output_shapes

:$N
truedivRealDivmul:z:0Cast:y:0*
T0*
_output_shapes

:$@
NegNegtruediv:z:0*
T0*
_output_shapes

:$D
RoundRoundtruediv:z:0*
T0*
_output_shapes

:$I
addAddV2Neg:y:0	Round:y:0*
T0*
_output_shapes

:$N
StopGradientStopGradientadd:z:0*
T0*
_output_shapes

:$[
add_1AddV2truediv:z:0StopGradient:output:0*
T0*
_output_shapes

:$\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Bv
clip_by_value/MinimumMinimum	add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:$T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �v
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes

:$R
mul_1MulCast:y:0clip_by_value:z:0*
T0*
_output_shapes

:$P
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C^
	truediv_1RealDiv	mul_1:z:0truediv_1/y:output:0*
T0*
_output_shapes

:$L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
mul_2Mulmul_2/x:output:0truediv_1:z:0*
T0*
_output_shapes

:$h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:$*
dtype0O
Neg_1NegReadVariableOp_1:value:0*
T0*
_output_shapes

:$M
add_2AddV2	Neg_1:y:0	mul_2:z:0*
T0*
_output_shapes

:$L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_3Mulmul_3/x:output:0	add_2:z:0*
T0*
_output_shapes

:$R
StopGradient_1StopGradient	mul_3:z:0*
T0*
_output_shapes

:$h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:$*
dtype0j
add_3AddV2ReadVariableOp_2:value:0StopGradient_1:output:0*
T0*
_output_shapes

:$;
ShapeShapeinputs*
T0*
_output_shapes
:Q
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
numX
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"   $   S
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:���������_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       c
	transpose	Transpose	add_3:z:0transpose/perm:output:0*
T0*
_output_shapes

:$`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ����f
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:$h
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:���������$S
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :dS
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :$�
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:v
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:���������d$I
Pow_1/xConst*
_output_shapes
: *
dtype0*
value	B :I
Pow_1/yConst*
_output_shapes
: *
dtype0*
value	B :Q
Pow_1PowPow_1/x:output:0Pow_1/y:output:0*
T0*
_output_shapes
: I
Cast_1Cast	Pow_1:z:0*

DstT0*

SrcT0*
_output_shapes
: f
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:$*
dtype0L
mul_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *   C]
mul_4MulReadVariableOp_3:value:0mul_4/y:output:0*
T0*
_output_shapes
:$P
	truediv_2RealDiv	mul_4:z:0
Cast_1:y:0*
T0*
_output_shapes
:$@
Neg_2Negtruediv_2:z:0*
T0*
_output_shapes
:$D
Round_1Roundtruediv_2:z:0*
T0*
_output_shapes
:$K
add_4AddV2	Neg_2:y:0Round_1:y:0*
T0*
_output_shapes
:$N
StopGradient_2StopGradient	add_4:z:0*
T0*
_output_shapes
:$[
add_5AddV2truediv_2:z:0StopGradient_2:output:0*
T0*
_output_shapes
:$^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �Bv
clip_by_value_1/MinimumMinimum	add_5:z:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:$V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   �x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:$R
mul_5Mul
Cast_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes
:$P
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   CZ
	truediv_3RealDiv	mul_5:z:0truediv_3/y:output:0*
T0*
_output_shapes
:$L
mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?R
mul_6Mulmul_6/x:output:0truediv_3:z:0*
T0*
_output_shapes
:$f
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource*
_output_shapes
:$*
dtype0K
Neg_3NegReadVariableOp_4:value:0*
T0*
_output_shapes
:$I
add_6AddV2	Neg_3:y:0	mul_6:z:0*
T0*
_output_shapes
:$L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?N
mul_7Mulmul_7/x:output:0	add_6:z:0*
T0*
_output_shapes
:$N
StopGradient_3StopGradient	mul_7:z:0*
T0*
_output_shapes
:$f
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource*
_output_shapes
:$*
dtype0f
add_7AddV2ReadVariableOp_5:value:0StopGradient_3:output:0*
T0*
_output_shapes
:$g
BiasAddBiasAddReshape_2:output:0	add_7:z:0*
T0*+
_output_shapes
:���������d$c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:���������d$�
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������d: : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_5:S O
+
_output_shapes
:���������d
 
_user_specified_nameinputs
�E
�

B__inference_model_layer_call_and_return_conditional_losses_4871607

input_cont

input_pxpy

input_cat0

input_cat1$
embedding0_4871552:$
embedding1_4871555:!
q_dense_4871560:
q_dense_4871562:)
batch_normalization_4871565:)
batch_normalization_4871567:)
batch_normalization_4871569:)
batch_normalization_4871571:#
q_dense_1_4871575:$
q_dense_1_4871577:$+
batch_normalization_1_4871580:$+
batch_normalization_1_4871582:$+
batch_normalization_1_4871584:$+
batch_normalization_1_4871586:$$
met_weight_4871590:$ 
met_weight_4871592:*
met_weight_minus_one_4871595:*
met_weight_minus_one_4871597:*
met_weight_minus_one_4871599:*
met_weight_minus_one_4871601:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�"embedding0/StatefulPartitionedCall�"embedding1/StatefulPartitionedCall�"met_weight/StatefulPartitionedCall�,met_weight_minus_one/StatefulPartitionedCall�q_dense/StatefulPartitionedCall�!q_dense_1/StatefulPartitionedCall�
"embedding0/StatefulPartitionedCallStatefulPartitionedCall
input_cat0embedding0_4871552*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_embedding0_layer_call_and_return_conditional_losses_4870710�
"embedding1/StatefulPartitionedCallStatefulPartitionedCall
input_cat1embedding1_4871555*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_embedding1_layer_call_and_return_conditional_losses_4870724�
concatenate/PartitionedCallPartitionedCall+embedding0/StatefulPartitionedCall:output:0+embedding1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_4870735�
concatenate_1/PartitionedCallPartitionedCall
input_cont$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_4870744�
q_dense/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0q_dense_4871560q_dense_4871562*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_q_dense_layer_call_and_return_conditional_losses_4870831�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall(q_dense/StatefulPartitionedCall:output:0batch_normalization_4871565batch_normalization_4871567batch_normalization_4871569batch_normalization_4871571*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_4870513�
q_activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_q_activation_layer_call_and_return_conditional_losses_4870895�
!q_dense_1/StatefulPartitionedCallStatefulPartitionedCall%q_activation/PartitionedCall:output:0q_dense_1_4871575q_dense_1_4871577*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_q_dense_1_layer_call_and_return_conditional_losses_4870982�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall*q_dense_1/StatefulPartitionedCall:output:0batch_normalization_1_4871580batch_normalization_1_4871582batch_normalization_1_4871584batch_normalization_1_4871586*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d$*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4870595�
q_activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_q_activation_1_layer_call_and_return_conditional_losses_4871046�
"met_weight/StatefulPartitionedCallStatefulPartitionedCall'q_activation_1/PartitionedCall:output:0met_weight_4871590met_weight_4871592*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_met_weight_layer_call_and_return_conditional_losses_4871133�
,met_weight_minus_one/StatefulPartitionedCallStatefulPartitionedCall+met_weight/StatefulPartitionedCall:output:0met_weight_minus_one_4871595met_weight_minus_one_4871597met_weight_minus_one_4871599met_weight_minus_one_4871601*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_met_weight_minus_one_layer_call_and_return_conditional_losses_4870663�
multiply/PartitionedCallPartitionedCall5met_weight_minus_one/StatefulPartitionedCall:output:0
input_pxpy*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_multiply_layer_call_and_return_conditional_losses_4871154�
output/PartitionedCallPartitionedCall!multiply/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_4870684n
IdentityIdentityoutput/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall#^embedding0/StatefulPartitionedCall#^embedding1/StatefulPartitionedCall#^met_weight/StatefulPartitionedCall-^met_weight_minus_one/StatefulPartitionedCall ^q_dense/StatefulPartitionedCall"^q_dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|:���������d:���������d:���������d:���������d: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2H
"embedding0/StatefulPartitionedCall"embedding0/StatefulPartitionedCall2H
"embedding1/StatefulPartitionedCall"embedding1/StatefulPartitionedCall2H
"met_weight/StatefulPartitionedCall"met_weight/StatefulPartitionedCall2\
,met_weight_minus_one/StatefulPartitionedCall,met_weight_minus_one/StatefulPartitionedCall2B
q_dense/StatefulPartitionedCallq_dense/StatefulPartitionedCall2F
!q_dense_1/StatefulPartitionedCall!q_dense_1/StatefulPartitionedCall:W S
+
_output_shapes
:���������d
$
_user_specified_name
input_cont:WS
+
_output_shapes
:���������d
$
_user_specified_name
input_pxpy:SO
'
_output_shapes
:���������d
$
_user_specified_name
input_cat0:SO
'
_output_shapes
:���������d
$
_user_specified_name
input_cat1
�
L
0__inference_q_activation_1_layer_call_fn_4873068

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������d$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_q_activation_1_layer_call_and_return_conditional_losses_4871046d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������d$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d$:S O
+
_output_shapes
:���������d$
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
A

input_cat03
serving_default_input_cat0:0���������d
A

input_cat13
serving_default_input_cat1:0���������d
E

input_cont7
serving_default_input_cont:0���������d
E

input_pxpy7
serving_default_input_pxpy:0���������d:
output0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer-15
layer-16
layer-17
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
�

embeddings
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses"
_tf_keras_layer
�
#
embeddings
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
�
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses"
_tf_keras_layer
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses"
_tf_keras_layer
�
6kernel_quantizer
6bias_quantizer
6kernel_quantizer_internal
6bias_quantizer_internal
7
quantizers

8kernel
9bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses"
_tf_keras_layer
�
@axis
	Agamma
Bbeta
Cmoving_mean
Dmoving_variance
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses"
_tf_keras_layer
�
K
activation
K	quantizer
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses"
_tf_keras_layer
�
6kernel_quantizer
6bias_quantizer
6kernel_quantizer_internal
6bias_quantizer_internal
R
quantizers

Skernel
Tbias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"
_tf_keras_layer
�
[axis
	\gamma
]beta
^moving_mean
_moving_variance
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses"
_tf_keras_layer
�
K
activation
K	quantizer
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses"
_tf_keras_layer
�
6kernel_quantizer
6bias_quantizer
6kernel_quantizer_internal
6bias_quantizer_internal
l
quantizers

mkernel
nbias
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses"
_tf_keras_layer
�
uaxis
	vgamma
wbeta
xmoving_mean
ymoving_variance
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�beta_1
�beta_2

�decay
�learning_rate
	�iterm�#m�8m�9m�Am�Bm�Sm�Tm�\m�]m�mm�nm�v�#v�8v�9v�Av�Bv�Sv�Tv�\v�]v�mv�nv�"
	optimizer
�
0
#1
82
93
A4
B5
C6
D7
S8
T9
\10
]11
^12
_13
m14
n15
v16
w17
x18
y19"
trackable_list_wrapper
v
0
#1
82
93
A4
B5
S6
T7
\8
]9
m10
n11"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
'__inference_model_layer_call_fn_4871201
'__inference_model_layer_call_fn_4871661
'__inference_model_layer_call_fn_4871709
'__inference_model_layer_call_fn_4871485�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
B__inference_model_layer_call_and_return_conditional_losses_4872116
B__inference_model_layer_call_and_return_conditional_losses_4872551
B__inference_model_layer_call_and_return_conditional_losses_4871546
B__inference_model_layer_call_and_return_conditional_losses_4871607�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
"__inference__wrapped_model_4870442
input_cont
input_pxpy
input_cat0
input_cat1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
-
�serving_default"
signature_map
':%2embedding0/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
�2�
,__inference_embedding0_layer_call_fn_4872608�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_embedding0_layer_call_and_return_conditional_losses_4872618�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
':%2embedding1/embeddings
'
#0"
trackable_list_wrapper
'
#0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
�2�
,__inference_embedding1_layer_call_fn_4872625�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_embedding1_layer_call_and_return_conditional_losses_4872635�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
�2�
-__inference_concatenate_layer_call_fn_4872641�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_concatenate_layer_call_and_return_conditional_losses_4872648�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
�2�
/__inference_concatenate_1_layer_call_fn_4872654�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_concatenate_1_layer_call_and_return_conditional_losses_4872661�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
.
60
61"
trackable_list_wrapper
 :2q_dense/kernel
:2q_dense/bias
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
�2�
)__inference_q_dense_layer_call_fn_4872670�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_q_dense_layer_call_and_return_conditional_losses_4872755�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
':%2batch_normalization/gamma
&:$2batch_normalization/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
<
A0
B1
C2
D3"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
�2�
5__inference_batch_normalization_layer_call_fn_4872768
5__inference_batch_normalization_layer_call_fn_4872781�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_4872801
P__inference_batch_normalization_layer_call_and_return_conditional_losses_4872835�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
�2�
.__inference_q_activation_layer_call_fn_4872840�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_q_activation_layer_call_and_return_conditional_losses_4872889�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
60
61"
trackable_list_wrapper
": $2q_dense_1/kernel
:$2q_dense_1/bias
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
�2�
+__inference_q_dense_1_layer_call_fn_4872898�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_q_dense_1_layer_call_and_return_conditional_losses_4872983�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
):'$2batch_normalization_1/gamma
(:&$2batch_normalization_1/beta
1:/$ (2!batch_normalization_1/moving_mean
5:3$ (2%batch_normalization_1/moving_variance
<
\0
]1
^2
_3"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
�2�
7__inference_batch_normalization_1_layer_call_fn_4872996
7__inference_batch_normalization_1_layer_call_fn_4873009�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4873029
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4873063�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
�2�
0__inference_q_activation_1_layer_call_fn_4873068�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
K__inference_q_activation_1_layer_call_and_return_conditional_losses_4873117�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
60
61"
trackable_list_wrapper
#:!$2met_weight/kernel
:2met_weight/bias
.
m0
n1"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
�2�
,__inference_met_weight_layer_call_fn_4873126�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_met_weight_layer_call_and_return_conditional_losses_4873211�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
(:&2met_weight_minus_one/gamma
':%2met_weight_minus_one/beta
0:. (2 met_weight_minus_one/moving_mean
4:2 (2$met_weight_minus_one/moving_variance
<
v0
w1
x2
y3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
6__inference_met_weight_minus_one_layer_call_fn_4873224
6__inference_met_weight_minus_one_layer_call_fn_4873237�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
Q__inference_met_weight_minus_one_layer_call_and_return_conditional_losses_4873257
Q__inference_met_weight_minus_one_layer_call_and_return_conditional_losses_4873277�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
*__inference_multiply_layer_call_fn_4873283�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_multiply_layer_call_and_return_conditional_losses_4873289�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
(__inference_output_layer_call_fn_4873294�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_output_layer_call_and_return_conditional_losses_4873300�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
X
C0
D1
^2
_3
v4
w5
x6
y7"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17"
trackable_list_wrapper
8
�0
�1
�2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_signature_wrapper_4872601
input_cat0
input_cat1
input_cont
input_pxpy"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
v0
w1
x2
y3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

�total

�count
�	variables
�	keras_api"
_tf_keras_metric
c

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"
_tf_keras_metric
c

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
,:*2Adam/embedding0/embeddings/m
,:*2Adam/embedding1/embeddings/m
%:#2Adam/q_dense/kernel/m
:2Adam/q_dense/bias/m
,:*2 Adam/batch_normalization/gamma/m
+:)2Adam/batch_normalization/beta/m
':%$2Adam/q_dense_1/kernel/m
!:$2Adam/q_dense_1/bias/m
.:,$2"Adam/batch_normalization_1/gamma/m
-:+$2!Adam/batch_normalization_1/beta/m
(:&$2Adam/met_weight/kernel/m
": 2Adam/met_weight/bias/m
,:*2Adam/embedding0/embeddings/v
,:*2Adam/embedding1/embeddings/v
%:#2Adam/q_dense/kernel/v
:2Adam/q_dense/bias/v
,:*2 Adam/batch_normalization/gamma/v
+:)2Adam/batch_normalization/beta/v
':%$2Adam/q_dense_1/kernel/v
!:$2Adam/q_dense_1/bias/v
.:,$2"Adam/batch_normalization_1/gamma/v
-:+$2!Adam/batch_normalization_1/beta/v
(:&$2Adam/met_weight/kernel/v
": 2Adam/met_weight/bias/v�
"__inference__wrapped_model_4870442�#89DACBST_\^]mnyvxw���
���
���
(�%

input_cont���������d
(�%

input_pxpy���������d
$�!

input_cat0���������d
$�!

input_cat1���������d
� "/�,
*
output �
output����������
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4873029|_\^]@�=
6�3
-�*
inputs������������������$
p 
� "2�/
(�%
0������������������$
� �
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4873063|^_\]@�=
6�3
-�*
inputs������������������$
p
� "2�/
(�%
0������������������$
� �
7__inference_batch_normalization_1_layer_call_fn_4872996o_\^]@�=
6�3
-�*
inputs������������������$
p 
� "%�"������������������$�
7__inference_batch_normalization_1_layer_call_fn_4873009o^_\]@�=
6�3
-�*
inputs������������������$
p
� "%�"������������������$�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_4872801|DACB@�=
6�3
-�*
inputs������������������
p 
� "2�/
(�%
0������������������
� �
P__inference_batch_normalization_layer_call_and_return_conditional_losses_4872835|CDAB@�=
6�3
-�*
inputs������������������
p
� "2�/
(�%
0������������������
� �
5__inference_batch_normalization_layer_call_fn_4872768oDACB@�=
6�3
-�*
inputs������������������
p 
� "%�"�������������������
5__inference_batch_normalization_layer_call_fn_4872781oCDAB@�=
6�3
-�*
inputs������������������
p
� "%�"�������������������
J__inference_concatenate_1_layer_call_and_return_conditional_losses_4872661�b�_
X�U
S�P
&�#
inputs/0���������d
&�#
inputs/1���������d
� ")�&
�
0���������d
� �
/__inference_concatenate_1_layer_call_fn_4872654�b�_
X�U
S�P
&�#
inputs/0���������d
&�#
inputs/1���������d
� "����������d�
H__inference_concatenate_layer_call_and_return_conditional_losses_4872648�b�_
X�U
S�P
&�#
inputs/0���������d
&�#
inputs/1���������d
� ")�&
�
0���������d
� �
-__inference_concatenate_layer_call_fn_4872641�b�_
X�U
S�P
&�#
inputs/0���������d
&�#
inputs/1���������d
� "����������d�
G__inference_embedding0_layer_call_and_return_conditional_losses_4872618_/�,
%�"
 �
inputs���������d
� ")�&
�
0���������d
� �
,__inference_embedding0_layer_call_fn_4872608R/�,
%�"
 �
inputs���������d
� "����������d�
G__inference_embedding1_layer_call_and_return_conditional_losses_4872635_#/�,
%�"
 �
inputs���������d
� ")�&
�
0���������d
� �
,__inference_embedding1_layer_call_fn_4872625R#/�,
%�"
 �
inputs���������d
� "����������d�
G__inference_met_weight_layer_call_and_return_conditional_losses_4873211dmn3�0
)�&
$�!
inputs���������d$
� ")�&
�
0���������d
� �
,__inference_met_weight_layer_call_fn_4873126Wmn3�0
)�&
$�!
inputs���������d$
� "����������d�
Q__inference_met_weight_minus_one_layer_call_and_return_conditional_losses_4873257|yvxw@�=
6�3
-�*
inputs������������������
p 
� "2�/
(�%
0������������������
� �
Q__inference_met_weight_minus_one_layer_call_and_return_conditional_losses_4873277|yvxw@�=
6�3
-�*
inputs������������������
p
� "2�/
(�%
0������������������
� �
6__inference_met_weight_minus_one_layer_call_fn_4873224oyvxw@�=
6�3
-�*
inputs������������������
p 
� "%�"�������������������
6__inference_met_weight_minus_one_layer_call_fn_4873237oyvxw@�=
6�3
-�*
inputs������������������
p
� "%�"�������������������
B__inference_model_layer_call_and_return_conditional_losses_4871546�#89DACBST_\^]mnyvxw���
���
���
(�%

input_cont���������d
(�%

input_pxpy���������d
$�!

input_cat0���������d
$�!

input_cat1���������d
p 

 
� "%�"
�
0���������
� �
B__inference_model_layer_call_and_return_conditional_losses_4871607�#89CDABST^_\]mnyvxw���
���
���
(�%

input_cont���������d
(�%

input_pxpy���������d
$�!

input_cat0���������d
$�!

input_cat1���������d
p

 
� "%�"
�
0���������
� �
B__inference_model_layer_call_and_return_conditional_losses_4872116�#89DACBST_\^]mnyvxw���
���
���
&�#
inputs/0���������d
&�#
inputs/1���������d
"�
inputs/2���������d
"�
inputs/3���������d
p 

 
� "%�"
�
0���������
� �
B__inference_model_layer_call_and_return_conditional_losses_4872551�#89CDABST^_\]mnyvxw���
���
���
&�#
inputs/0���������d
&�#
inputs/1���������d
"�
inputs/2���������d
"�
inputs/3���������d
p

 
� "%�"
�
0���������
� �
'__inference_model_layer_call_fn_4871201�#89DACBST_\^]mnyvxw���
���
���
(�%

input_cont���������d
(�%

input_pxpy���������d
$�!

input_cat0���������d
$�!

input_cat1���������d
p 

 
� "�����������
'__inference_model_layer_call_fn_4871485�#89CDABST^_\]mnyvxw���
���
���
(�%

input_cont���������d
(�%

input_pxpy���������d
$�!

input_cat0���������d
$�!

input_cat1���������d
p

 
� "�����������
'__inference_model_layer_call_fn_4871661�#89DACBST_\^]mnyvxw���
���
���
&�#
inputs/0���������d
&�#
inputs/1���������d
"�
inputs/2���������d
"�
inputs/3���������d
p 

 
� "�����������
'__inference_model_layer_call_fn_4871709�#89CDABST^_\]mnyvxw���
���
���
&�#
inputs/0���������d
&�#
inputs/1���������d
"�
inputs/2���������d
"�
inputs/3���������d
p

 
� "�����������
E__inference_multiply_layer_call_and_return_conditional_losses_4873289�b�_
X�U
S�P
&�#
inputs/0���������d
&�#
inputs/1���������d
� ")�&
�
0���������d
� �
*__inference_multiply_layer_call_fn_4873283�b�_
X�U
S�P
&�#
inputs/0���������d
&�#
inputs/1���������d
� "����������d�
C__inference_output_layer_call_and_return_conditional_losses_4873300{I�F
?�<
6�3
inputs'���������������������������

 
� ".�+
$�!
0������������������
� �
(__inference_output_layer_call_fn_4873294nI�F
?�<
6�3
inputs'���������������������������

 
� "!��������������������
K__inference_q_activation_1_layer_call_and_return_conditional_losses_4873117`3�0
)�&
$�!
inputs���������d$
� ")�&
�
0���������d$
� �
0__inference_q_activation_1_layer_call_fn_4873068S3�0
)�&
$�!
inputs���������d$
� "����������d$�
I__inference_q_activation_layer_call_and_return_conditional_losses_4872889`3�0
)�&
$�!
inputs���������d
� ")�&
�
0���������d
� �
.__inference_q_activation_layer_call_fn_4872840S3�0
)�&
$�!
inputs���������d
� "����������d�
F__inference_q_dense_1_layer_call_and_return_conditional_losses_4872983dST3�0
)�&
$�!
inputs���������d
� ")�&
�
0���������d$
� �
+__inference_q_dense_1_layer_call_fn_4872898WST3�0
)�&
$�!
inputs���������d
� "����������d$�
D__inference_q_dense_layer_call_and_return_conditional_losses_4872755d893�0
)�&
$�!
inputs���������d
� ")�&
�
0���������d
� �
)__inference_q_dense_layer_call_fn_4872670W893�0
)�&
$�!
inputs���������d
� "����������d�
%__inference_signature_wrapper_4872601�#89DACBST_\^]mnyvxw���
� 
���
2

input_cat0$�!

input_cat0���������d
2

input_cat1$�!

input_cat1���������d
6

input_cont(�%

input_cont���������d
6

input_pxpy(�%

input_pxpy���������d"/�,
*
output �
output���������