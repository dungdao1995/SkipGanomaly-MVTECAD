̓#
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
@
ReadVariableOp
resource
value"dtype"
dtypetype?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.12v2.7.0-217-g2a0f59ecfe68ͳ
?
discriminator/conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*/
shared_name discriminator/conv2d_10/kernel
?
2discriminator/conv2d_10/kernel/Read/ReadVariableOpReadVariableOpdiscriminator/conv2d_10/kernel*&
_output_shapes
:d*
dtype0
?
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:@*
dtype0
?
batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_9/gamma
?
/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_9/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_9/beta
?
.batch_normalization_9/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_9/beta*
_output_shapes
:@*
dtype0
?
!batch_normalization_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_9/moving_mean
?
5batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_9/moving_mean*
_output_shapes
:@*
dtype0
?
%batch_normalization_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_9/moving_variance
?
9batch_normalization_9/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_9/moving_variance*
_output_shapes
:@*
dtype0
?
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?* 
shared_nameconv2d_6/kernel
|
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*'
_output_shapes
:@?*
dtype0
?
batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_10/gamma
?
0batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_10/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_10/beta
?
/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_10/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_10/moving_mean
?
6batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_10/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_10/moving_variance
?
:batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_10/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??* 
shared_nameconv2d_7/kernel
}
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*(
_output_shapes
:??*
dtype0
?
batch_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_11/gamma
?
0batch_normalization_11/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_11/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_11/beta
?
/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_11/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_11/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_11/moving_mean
?
6batch_normalization_11/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_11/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_11/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_11/moving_variance
?
:batch_normalization_11/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_11/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??* 
shared_nameconv2d_8/kernel
}
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*(
_output_shapes
:??*
dtype0
?
batch_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_12/gamma
?
0batch_normalization_12/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_12/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_12/beta
?
/batch_normalization_12/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_12/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_12/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_12/moving_mean
?
6batch_normalization_12/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_12/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_12/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_12/moving_variance
?
:batch_normalization_12/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_12/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?d* 
shared_nameconv2d_9/kernel
|
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*'
_output_shapes
:?d*
dtype0
?
batch_normalization_13/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*-
shared_namebatch_normalization_13/gamma
?
0batch_normalization_13/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_13/gamma*
_output_shapes
:d*
dtype0
?
batch_normalization_13/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*,
shared_namebatch_normalization_13/beta
?
/batch_normalization_13/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_13/beta*
_output_shapes
:d*
dtype0
?
"batch_normalization_13/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*3
shared_name$"batch_normalization_13/moving_mean
?
6batch_normalization_13/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_13/moving_mean*
_output_shapes
:d*
dtype0
?
&batch_normalization_13/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*7
shared_name(&batch_normalization_13/moving_variance
?
:batch_normalization_13/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_13/moving_variance*
_output_shapes
:d*
dtype0

NoOpNoOp
?d
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?c
value?cB?c B?c
?
	encoder_1
	encoder_2
	encoder_3
	encoder_4

center
outputs
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
b

conv_layer
	variables
trainable_variables
regularization_losses
	keras_api
b

conv_layer
	variables
trainable_variables
regularization_losses
	keras_api
b

conv_layer
	variables
trainable_variables
regularization_losses
	keras_api
b

conv_layer
	variables
trainable_variables
regularization_losses
	keras_api
b
 
conv_layer
!	variables
"trainable_variables
#regularization_losses
$	keras_api
^

%kernel
&	variables
'trainable_variables
(regularization_losses
)	keras_api
?
*0
+1
,2
-3
.4
/5
06
17
28
39
410
511
612
713
814
915
:16
;17
<18
=19
>20
?21
@22
A23
B24
%25
v
*0
+1
,2
/3
04
15
46
57
68
99
:10
;11
>12
?13
@14
%15
 
?
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
	regularization_losses
 
?
Hlayer_with_weights-0
Hlayer-0
Ilayer_with_weights-1
Ilayer-1
Jlayer-2
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
#
*0
+1
,2
-3
.4

*0
+1
,2
 
?
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
?
Tlayer_with_weights-0
Tlayer-0
Ulayer_with_weights-1
Ulayer-1
Vlayer-2
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
#
/0
01
12
23
34

/0
01
12
 
?
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
?
`layer_with_weights-0
`layer-0
alayer_with_weights-1
alayer-1
blayer-2
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
#
40
51
62
73
84

40
51
62
 
?
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
	variables
trainable_variables
regularization_losses
?
llayer_with_weights-0
llayer-0
mlayer_with_weights-1
mlayer-1
nlayer-2
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
#
90
:1
;2
<3
=4

90
:1
;2
 
?
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
	variables
trainable_variables
regularization_losses
?
xlayer_with_weights-0
xlayer-0
ylayer_with_weights-1
ylayer-1
zlayer-2
{	variables
|trainable_variables
}regularization_losses
~	keras_api
#
>0
?1
@2
A3
B4

>0
?1
@2
 
?
non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
!	variables
"trainable_variables
#regularization_losses
][
VARIABLE_VALUEdiscriminator/conv2d_10/kernel)outputs/kernel/.ATTRIBUTES/VARIABLE_VALUE

%0

%0
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
&	variables
'trainable_variables
(regularization_losses
KI
VARIABLE_VALUEconv2d_5/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_9/gamma&variables/1/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEbatch_normalization_9/beta&variables/2/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!batch_normalization_9/moving_mean&variables/3/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%batch_normalization_9/moving_variance&variables/4/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_6/kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_10/gamma&variables/6/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_10/beta&variables/7/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"batch_normalization_10/moving_mean&variables/8/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&batch_normalization_10/moving_variance&variables/9/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_7/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_normalization_11/gamma'variables/11/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_11/beta'variables/12/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_11/moving_mean'variables/13/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_11/moving_variance'variables/14/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_8/kernel'variables/15/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_normalization_12/gamma'variables/16/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_12/beta'variables/17/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_12/moving_mean'variables/18/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_12/moving_variance'variables/19/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_9/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_normalization_13/gamma'variables/21/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_13/beta'variables/22/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_13/moving_mean'variables/23/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_13/moving_variance'variables/24/.ATTRIBUTES/VARIABLE_VALUE
F
-0
.1
22
33
74
85
<6
=7
A8
B9
*
0
1
2
3
4
5
 
 
 
b

*kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis
	+gamma
,beta
-moving_mean
.moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
#
*0
+1
,2
-3
.4

*0
+1
,2
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses

-0
.1

0
 
 
 
b

/kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis
	0gamma
1beta
2moving_mean
3moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
#
/0
01
12
23
34

/0
01
12
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses

20
31

0
 
 
 
b

4kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis
	5gamma
6beta
7moving_mean
8moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
#
40
51
62
73
84

40
51
62
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
c	variables
dtrainable_variables
eregularization_losses

70
81

0
 
 
 
b

9kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis
	:gamma
;beta
<moving_mean
=moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
#
90
:1
;2
<3
=4

90
:1
;2
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
o	variables
ptrainable_variables
qregularization_losses

<0
=1

0
 
 
 
b

>kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis
	?gamma
@beta
Amoving_mean
Bmoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
#
>0
?1
@2
A3
B4

>0
?1
@2
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
{	variables
|trainable_variables
}regularization_losses

A0
B1

 0
 
 
 
 
 
 
 
 

*0

*0
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 

+0
,1
-2
.3

+0
,1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

-0
.1

H0
I1
J2
 
 
 

/0

/0
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 

00
11
22
33

00
11
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

20
31

T0
U1
V2
 
 
 

40

40
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 

50
61
72
83

50
61
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

70
81

`0
a1
b2
 
 
 

90

90
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 

:0
;1
<2
=3

:0
;1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

<0
=1

l0
m1
n2
 
 
 

>0

>0
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 

?0
@1
A2
B3

?0
@1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses

A0
B1

x0
y1
z2
 
 
 
 
 
 
 
 

-0
.1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

20
31
 
 
 
 
 
 
 
 
 
 
 
 
 
 

70
81
 
 
 
 
 
 
 
 
 
 
 
 
 
 

<0
=1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

A0
B1
 
 
 
 
 
 
 
 
 
?
serving_default_input_1Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_5/kernelbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_6/kernelbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv2d_7/kernelbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_varianceconv2d_8/kernelbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_varianceconv2d_9/kernelbatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_variancediscriminator/conv2d_10/kernel*&
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:?????????:?????????d*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_969539
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename2discriminator/conv2d_10/kernel/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp/batch_normalization_9/gamma/Read/ReadVariableOp.batch_normalization_9/beta/Read/ReadVariableOp5batch_normalization_9/moving_mean/Read/ReadVariableOp9batch_normalization_9/moving_variance/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp0batch_normalization_10/gamma/Read/ReadVariableOp/batch_normalization_10/beta/Read/ReadVariableOp6batch_normalization_10/moving_mean/Read/ReadVariableOp:batch_normalization_10/moving_variance/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp0batch_normalization_11/gamma/Read/ReadVariableOp/batch_normalization_11/beta/Read/ReadVariableOp6batch_normalization_11/moving_mean/Read/ReadVariableOp:batch_normalization_11/moving_variance/Read/ReadVariableOp#conv2d_8/kernel/Read/ReadVariableOp0batch_normalization_12/gamma/Read/ReadVariableOp/batch_normalization_12/beta/Read/ReadVariableOp6batch_normalization_12/moving_mean/Read/ReadVariableOp:batch_normalization_12/moving_variance/Read/ReadVariableOp#conv2d_9/kernel/Read/ReadVariableOp0batch_normalization_13/gamma/Read/ReadVariableOp/batch_normalization_13/beta/Read/ReadVariableOp6batch_normalization_13/moving_mean/Read/ReadVariableOp:batch_normalization_13/moving_variance/Read/ReadVariableOpConst*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_971452
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamediscriminator/conv2d_10/kernelconv2d_5/kernelbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_varianceconv2d_6/kernelbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv2d_7/kernelbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_varianceconv2d_8/kernelbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_varianceconv2d_9/kernelbatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_variance*&
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_971540??
?
?
I__inference_sequential_13_layer_call_and_return_conditional_losses_968682

inputs*
conv2d_9_968644:?d+
batch_normalization_13_968665:d+
batch_normalization_13_968667:d+
batch_normalization_13_968669:d+
batch_normalization_13_968671:d
identity??.batch_normalization_13/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_9_968644*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_968643?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0batch_normalization_13_968665batch_normalization_13_968667batch_normalization_13_968669batch_normalization_13_968671*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_968664?
leaky_re_lu_9/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_968679}
IdentityIdentity&leaky_re_lu_9/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????d?
NoOpNoOp/^batch_normalization_13/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?(
I__inference_discriminator_layer_call_and_return_conditional_losses_969855

inputs[
Aconv_block_5_sequential_9_conv2d_5_conv2d_readvariableop_resource:@U
Gconv_block_5_sequential_9_batch_normalization_9_readvariableop_resource:@W
Iconv_block_5_sequential_9_batch_normalization_9_readvariableop_1_resource:@f
Xconv_block_5_sequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:@h
Zconv_block_5_sequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:@]
Bconv_block_6_sequential_10_conv2d_6_conv2d_readvariableop_resource:@?X
Iconv_block_6_sequential_10_batch_normalization_10_readvariableop_resource:	?Z
Kconv_block_6_sequential_10_batch_normalization_10_readvariableop_1_resource:	?i
Zconv_block_6_sequential_10_batch_normalization_10_fusedbatchnormv3_readvariableop_resource:	?k
\conv_block_6_sequential_10_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:	?^
Bconv_block_7_sequential_11_conv2d_7_conv2d_readvariableop_resource:??X
Iconv_block_7_sequential_11_batch_normalization_11_readvariableop_resource:	?Z
Kconv_block_7_sequential_11_batch_normalization_11_readvariableop_1_resource:	?i
Zconv_block_7_sequential_11_batch_normalization_11_fusedbatchnormv3_readvariableop_resource:	?k
\conv_block_7_sequential_11_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:	?^
Bconv_block_8_sequential_12_conv2d_8_conv2d_readvariableop_resource:??X
Iconv_block_8_sequential_12_batch_normalization_12_readvariableop_resource:	?Z
Kconv_block_8_sequential_12_batch_normalization_12_readvariableop_1_resource:	?i
Zconv_block_8_sequential_12_batch_normalization_12_fusedbatchnormv3_readvariableop_resource:	?k
\conv_block_8_sequential_12_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:	?]
Bconv_block_9_sequential_13_conv2d_9_conv2d_readvariableop_resource:?dW
Iconv_block_9_sequential_13_batch_normalization_13_readvariableop_resource:dY
Kconv_block_9_sequential_13_batch_normalization_13_readvariableop_1_resource:dh
Zconv_block_9_sequential_13_batch_normalization_13_fusedbatchnormv3_readvariableop_resource:dj
\conv_block_9_sequential_13_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:dB
(conv2d_10_conv2d_readvariableop_resource:d
identity

identity_1??conv2d_10/Conv2D/ReadVariableOp?>conv_block_5/sequential_9/batch_normalization_9/AssignNewValue?@conv_block_5/sequential_9/batch_normalization_9/AssignNewValue_1?Oconv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp?Qconv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?>conv_block_5/sequential_9/batch_normalization_9/ReadVariableOp?@conv_block_5/sequential_9/batch_normalization_9/ReadVariableOp_1?8conv_block_5/sequential_9/conv2d_5/Conv2D/ReadVariableOp?@conv_block_6/sequential_10/batch_normalization_10/AssignNewValue?Bconv_block_6/sequential_10/batch_normalization_10/AssignNewValue_1?Qconv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp?Sconv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?@conv_block_6/sequential_10/batch_normalization_10/ReadVariableOp?Bconv_block_6/sequential_10/batch_normalization_10/ReadVariableOp_1?9conv_block_6/sequential_10/conv2d_6/Conv2D/ReadVariableOp?@conv_block_7/sequential_11/batch_normalization_11/AssignNewValue?Bconv_block_7/sequential_11/batch_normalization_11/AssignNewValue_1?Qconv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp?Sconv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1?@conv_block_7/sequential_11/batch_normalization_11/ReadVariableOp?Bconv_block_7/sequential_11/batch_normalization_11/ReadVariableOp_1?9conv_block_7/sequential_11/conv2d_7/Conv2D/ReadVariableOp?@conv_block_8/sequential_12/batch_normalization_12/AssignNewValue?Bconv_block_8/sequential_12/batch_normalization_12/AssignNewValue_1?Qconv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp?Sconv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?@conv_block_8/sequential_12/batch_normalization_12/ReadVariableOp?Bconv_block_8/sequential_12/batch_normalization_12/ReadVariableOp_1?9conv_block_8/sequential_12/conv2d_8/Conv2D/ReadVariableOp?@conv_block_9/sequential_13/batch_normalization_13/AssignNewValue?Bconv_block_9/sequential_13/batch_normalization_13/AssignNewValue_1?Qconv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp?Sconv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?@conv_block_9/sequential_13/batch_normalization_13/ReadVariableOp?Bconv_block_9/sequential_13/batch_normalization_13/ReadVariableOp_1?9conv_block_9/sequential_13/conv2d_9/Conv2D/ReadVariableOp?
8conv_block_5/sequential_9/conv2d_5/Conv2D/ReadVariableOpReadVariableOpAconv_block_5_sequential_9_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
)conv_block_5/sequential_9/conv2d_5/Conv2DConv2Dinputs@conv_block_5/sequential_9/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
>conv_block_5/sequential_9/batch_normalization_9/ReadVariableOpReadVariableOpGconv_block_5_sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:@*
dtype0?
@conv_block_5/sequential_9/batch_normalization_9/ReadVariableOp_1ReadVariableOpIconv_block_5_sequential_9_batch_normalization_9_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
Oconv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpXconv_block_5_sequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
Qconv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpZconv_block_5_sequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
@conv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV32conv_block_5/sequential_9/conv2d_5/Conv2D:output:0Fconv_block_5/sequential_9/batch_normalization_9/ReadVariableOp:value:0Hconv_block_5/sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Wconv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Yconv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
>conv_block_5/sequential_9/batch_normalization_9/AssignNewValueAssignVariableOpXconv_block_5_sequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resourceMconv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3:batch_mean:0P^conv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
@conv_block_5/sequential_9/batch_normalization_9/AssignNewValue_1AssignVariableOpZconv_block_5_sequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resourceQconv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3:batch_variance:0R^conv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
1conv_block_5/sequential_9/leaky_re_lu_5/LeakyRelu	LeakyReluDconv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@@@?
9conv_block_6/sequential_10/conv2d_6/Conv2D/ReadVariableOpReadVariableOpBconv_block_6_sequential_10_conv2d_6_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
*conv_block_6/sequential_10/conv2d_6/Conv2DConv2D?conv_block_5/sequential_9/leaky_re_lu_5/LeakyRelu:activations:0Aconv_block_6/sequential_10/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
@conv_block_6/sequential_10/batch_normalization_10/ReadVariableOpReadVariableOpIconv_block_6_sequential_10_batch_normalization_10_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Bconv_block_6/sequential_10/batch_normalization_10/ReadVariableOp_1ReadVariableOpKconv_block_6_sequential_10_batch_normalization_10_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Qconv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpZconv_block_6_sequential_10_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Sconv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp\conv_block_6_sequential_10_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Bconv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3FusedBatchNormV33conv_block_6/sequential_10/conv2d_6/Conv2D:output:0Hconv_block_6/sequential_10/batch_normalization_10/ReadVariableOp:value:0Jconv_block_6/sequential_10/batch_normalization_10/ReadVariableOp_1:value:0Yconv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0[conv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
@conv_block_6/sequential_10/batch_normalization_10/AssignNewValueAssignVariableOpZconv_block_6_sequential_10_batch_normalization_10_fusedbatchnormv3_readvariableop_resourceOconv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3:batch_mean:0R^conv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
Bconv_block_6/sequential_10/batch_normalization_10/AssignNewValue_1AssignVariableOp\conv_block_6_sequential_10_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resourceSconv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3:batch_variance:0T^conv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
2conv_block_6/sequential_10/leaky_re_lu_6/LeakyRelu	LeakyReluFconv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3:y:0*0
_output_shapes
:?????????  ??
9conv_block_7/sequential_11/conv2d_7/Conv2D/ReadVariableOpReadVariableOpBconv_block_7_sequential_11_conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
*conv_block_7/sequential_11/conv2d_7/Conv2DConv2D@conv_block_6/sequential_10/leaky_re_lu_6/LeakyRelu:activations:0Aconv_block_7/sequential_11/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
@conv_block_7/sequential_11/batch_normalization_11/ReadVariableOpReadVariableOpIconv_block_7_sequential_11_batch_normalization_11_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Bconv_block_7/sequential_11/batch_normalization_11/ReadVariableOp_1ReadVariableOpKconv_block_7_sequential_11_batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Qconv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpZconv_block_7_sequential_11_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Sconv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp\conv_block_7_sequential_11_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Bconv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3FusedBatchNormV33conv_block_7/sequential_11/conv2d_7/Conv2D:output:0Hconv_block_7/sequential_11/batch_normalization_11/ReadVariableOp:value:0Jconv_block_7/sequential_11/batch_normalization_11/ReadVariableOp_1:value:0Yconv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0[conv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
@conv_block_7/sequential_11/batch_normalization_11/AssignNewValueAssignVariableOpZconv_block_7_sequential_11_batch_normalization_11_fusedbatchnormv3_readvariableop_resourceOconv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3:batch_mean:0R^conv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
Bconv_block_7/sequential_11/batch_normalization_11/AssignNewValue_1AssignVariableOp\conv_block_7_sequential_11_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resourceSconv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3:batch_variance:0T^conv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
2conv_block_7/sequential_11/leaky_re_lu_7/LeakyRelu	LeakyReluFconv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3:y:0*0
_output_shapes
:???????????
9conv_block_8/sequential_12/conv2d_8/Conv2D/ReadVariableOpReadVariableOpBconv_block_8_sequential_12_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
*conv_block_8/sequential_12/conv2d_8/Conv2DConv2D@conv_block_7/sequential_11/leaky_re_lu_7/LeakyRelu:activations:0Aconv_block_8/sequential_12/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
@conv_block_8/sequential_12/batch_normalization_12/ReadVariableOpReadVariableOpIconv_block_8_sequential_12_batch_normalization_12_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Bconv_block_8/sequential_12/batch_normalization_12/ReadVariableOp_1ReadVariableOpKconv_block_8_sequential_12_batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Qconv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpZconv_block_8_sequential_12_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Sconv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp\conv_block_8_sequential_12_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Bconv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3FusedBatchNormV33conv_block_8/sequential_12/conv2d_8/Conv2D:output:0Hconv_block_8/sequential_12/batch_normalization_12/ReadVariableOp:value:0Jconv_block_8/sequential_12/batch_normalization_12/ReadVariableOp_1:value:0Yconv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0[conv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
@conv_block_8/sequential_12/batch_normalization_12/AssignNewValueAssignVariableOpZconv_block_8_sequential_12_batch_normalization_12_fusedbatchnormv3_readvariableop_resourceOconv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3:batch_mean:0R^conv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
Bconv_block_8/sequential_12/batch_normalization_12/AssignNewValue_1AssignVariableOp\conv_block_8_sequential_12_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resourceSconv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3:batch_variance:0T^conv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
2conv_block_8/sequential_12/leaky_re_lu_8/LeakyRelu	LeakyReluFconv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3:y:0*0
_output_shapes
:???????????
9conv_block_9/sequential_13/conv2d_9/Conv2D/ReadVariableOpReadVariableOpBconv_block_9_sequential_13_conv2d_9_conv2d_readvariableop_resource*'
_output_shapes
:?d*
dtype0?
*conv_block_9/sequential_13/conv2d_9/Conv2DConv2D@conv_block_8/sequential_12/leaky_re_lu_8/LeakyRelu:activations:0Aconv_block_9/sequential_13/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d*
paddingSAME*
strides
?
@conv_block_9/sequential_13/batch_normalization_13/ReadVariableOpReadVariableOpIconv_block_9_sequential_13_batch_normalization_13_readvariableop_resource*
_output_shapes
:d*
dtype0?
Bconv_block_9/sequential_13/batch_normalization_13/ReadVariableOp_1ReadVariableOpKconv_block_9_sequential_13_batch_normalization_13_readvariableop_1_resource*
_output_shapes
:d*
dtype0?
Qconv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpZconv_block_9_sequential_13_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:d*
dtype0?
Sconv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp\conv_block_9_sequential_13_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:d*
dtype0?
Bconv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3FusedBatchNormV33conv_block_9/sequential_13/conv2d_9/Conv2D:output:0Hconv_block_9/sequential_13/batch_normalization_13/ReadVariableOp:value:0Jconv_block_9/sequential_13/batch_normalization_13/ReadVariableOp_1:value:0Yconv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0[conv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????d:d:d:d:d:*
epsilon%o?:*
exponential_avg_factor%
?#<?
@conv_block_9/sequential_13/batch_normalization_13/AssignNewValueAssignVariableOpZconv_block_9_sequential_13_batch_normalization_13_fusedbatchnormv3_readvariableop_resourceOconv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3:batch_mean:0R^conv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
Bconv_block_9/sequential_13/batch_normalization_13/AssignNewValue_1AssignVariableOp\conv_block_9_sequential_13_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resourceSconv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3:batch_variance:0T^conv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
2conv_block_9/sequential_13/leaky_re_lu_9/LeakyRelu	LeakyReluFconv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3:y:0*/
_output_shapes
:?????????d?
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:d*
dtype0?
conv2d_10/Conv2DConv2D@conv_block_9/sequential_13/leaky_re_lu_9/LeakyRelu:activations:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
q
conv2d_10/SigmoidSigmoidconv2d_10/Conv2D:output:0*
T0*/
_output_shapes
:?????????l
IdentityIdentityconv2d_10/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:??????????

Identity_1Identity@conv_block_9/sequential_13/leaky_re_lu_9/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????d?
NoOpNoOp ^conv2d_10/Conv2D/ReadVariableOp?^conv_block_5/sequential_9/batch_normalization_9/AssignNewValueA^conv_block_5/sequential_9/batch_normalization_9/AssignNewValue_1P^conv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpR^conv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?^conv_block_5/sequential_9/batch_normalization_9/ReadVariableOpA^conv_block_5/sequential_9/batch_normalization_9/ReadVariableOp_19^conv_block_5/sequential_9/conv2d_5/Conv2D/ReadVariableOpA^conv_block_6/sequential_10/batch_normalization_10/AssignNewValueC^conv_block_6/sequential_10/batch_normalization_10/AssignNewValue_1R^conv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOpT^conv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1A^conv_block_6/sequential_10/batch_normalization_10/ReadVariableOpC^conv_block_6/sequential_10/batch_normalization_10/ReadVariableOp_1:^conv_block_6/sequential_10/conv2d_6/Conv2D/ReadVariableOpA^conv_block_7/sequential_11/batch_normalization_11/AssignNewValueC^conv_block_7/sequential_11/batch_normalization_11/AssignNewValue_1R^conv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOpT^conv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1A^conv_block_7/sequential_11/batch_normalization_11/ReadVariableOpC^conv_block_7/sequential_11/batch_normalization_11/ReadVariableOp_1:^conv_block_7/sequential_11/conv2d_7/Conv2D/ReadVariableOpA^conv_block_8/sequential_12/batch_normalization_12/AssignNewValueC^conv_block_8/sequential_12/batch_normalization_12/AssignNewValue_1R^conv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOpT^conv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1A^conv_block_8/sequential_12/batch_normalization_12/ReadVariableOpC^conv_block_8/sequential_12/batch_normalization_12/ReadVariableOp_1:^conv_block_8/sequential_12/conv2d_8/Conv2D/ReadVariableOpA^conv_block_9/sequential_13/batch_normalization_13/AssignNewValueC^conv_block_9/sequential_13/batch_normalization_13/AssignNewValue_1R^conv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOpT^conv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1A^conv_block_9/sequential_13/batch_normalization_13/ReadVariableOpC^conv_block_9/sequential_13/batch_normalization_13/ReadVariableOp_1:^conv_block_9/sequential_13/conv2d_9/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2?
>conv_block_5/sequential_9/batch_normalization_9/AssignNewValue>conv_block_5/sequential_9/batch_normalization_9/AssignNewValue2?
@conv_block_5/sequential_9/batch_normalization_9/AssignNewValue_1@conv_block_5/sequential_9/batch_normalization_9/AssignNewValue_12?
Oconv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpOconv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2?
Qconv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Qconv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12?
>conv_block_5/sequential_9/batch_normalization_9/ReadVariableOp>conv_block_5/sequential_9/batch_normalization_9/ReadVariableOp2?
@conv_block_5/sequential_9/batch_normalization_9/ReadVariableOp_1@conv_block_5/sequential_9/batch_normalization_9/ReadVariableOp_12t
8conv_block_5/sequential_9/conv2d_5/Conv2D/ReadVariableOp8conv_block_5/sequential_9/conv2d_5/Conv2D/ReadVariableOp2?
@conv_block_6/sequential_10/batch_normalization_10/AssignNewValue@conv_block_6/sequential_10/batch_normalization_10/AssignNewValue2?
Bconv_block_6/sequential_10/batch_normalization_10/AssignNewValue_1Bconv_block_6/sequential_10/batch_normalization_10/AssignNewValue_12?
Qconv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOpQconv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp2?
Sconv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1Sconv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12?
@conv_block_6/sequential_10/batch_normalization_10/ReadVariableOp@conv_block_6/sequential_10/batch_normalization_10/ReadVariableOp2?
Bconv_block_6/sequential_10/batch_normalization_10/ReadVariableOp_1Bconv_block_6/sequential_10/batch_normalization_10/ReadVariableOp_12v
9conv_block_6/sequential_10/conv2d_6/Conv2D/ReadVariableOp9conv_block_6/sequential_10/conv2d_6/Conv2D/ReadVariableOp2?
@conv_block_7/sequential_11/batch_normalization_11/AssignNewValue@conv_block_7/sequential_11/batch_normalization_11/AssignNewValue2?
Bconv_block_7/sequential_11/batch_normalization_11/AssignNewValue_1Bconv_block_7/sequential_11/batch_normalization_11/AssignNewValue_12?
Qconv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOpQconv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp2?
Sconv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1Sconv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12?
@conv_block_7/sequential_11/batch_normalization_11/ReadVariableOp@conv_block_7/sequential_11/batch_normalization_11/ReadVariableOp2?
Bconv_block_7/sequential_11/batch_normalization_11/ReadVariableOp_1Bconv_block_7/sequential_11/batch_normalization_11/ReadVariableOp_12v
9conv_block_7/sequential_11/conv2d_7/Conv2D/ReadVariableOp9conv_block_7/sequential_11/conv2d_7/Conv2D/ReadVariableOp2?
@conv_block_8/sequential_12/batch_normalization_12/AssignNewValue@conv_block_8/sequential_12/batch_normalization_12/AssignNewValue2?
Bconv_block_8/sequential_12/batch_normalization_12/AssignNewValue_1Bconv_block_8/sequential_12/batch_normalization_12/AssignNewValue_12?
Qconv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOpQconv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp2?
Sconv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1Sconv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12?
@conv_block_8/sequential_12/batch_normalization_12/ReadVariableOp@conv_block_8/sequential_12/batch_normalization_12/ReadVariableOp2?
Bconv_block_8/sequential_12/batch_normalization_12/ReadVariableOp_1Bconv_block_8/sequential_12/batch_normalization_12/ReadVariableOp_12v
9conv_block_8/sequential_12/conv2d_8/Conv2D/ReadVariableOp9conv_block_8/sequential_12/conv2d_8/Conv2D/ReadVariableOp2?
@conv_block_9/sequential_13/batch_normalization_13/AssignNewValue@conv_block_9/sequential_13/batch_normalization_13/AssignNewValue2?
Bconv_block_9/sequential_13/batch_normalization_13/AssignNewValue_1Bconv_block_9/sequential_13/batch_normalization_13/AssignNewValue_12?
Qconv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOpQconv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp2?
Sconv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1Sconv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12?
@conv_block_9/sequential_13/batch_normalization_13/ReadVariableOp@conv_block_9/sequential_13/batch_normalization_13/ReadVariableOp2?
Bconv_block_9/sequential_13/batch_normalization_13/ReadVariableOp_1Bconv_block_9/sequential_13/batch_normalization_13/ReadVariableOp_12v
9conv_block_9/sequential_13/conv2d_9/Conv2D/ReadVariableOp9conv_block_9/sequential_13/conv2d_9/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
-__inference_conv_block_7_layer_call_fn_968069
input_1#
unknown:??
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_7_layer_call_and_return_conditional_losses_968056x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????  ?: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:?????????  ?
!
_user_specified_name	input_1
?
?
H__inference_conv_block_5_layer_call_and_return_conditional_losses_969907

inputsN
4sequential_9_conv2d_5_conv2d_readvariableop_resource:@H
:sequential_9_batch_normalization_9_readvariableop_resource:@J
<sequential_9_batch_normalization_9_readvariableop_1_resource:@Y
Ksequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:@[
Msequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:@
identity??Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp?Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?1sequential_9/batch_normalization_9/ReadVariableOp?3sequential_9/batch_normalization_9/ReadVariableOp_1?+sequential_9/conv2d_5/Conv2D/ReadVariableOp?
+sequential_9/conv2d_5/Conv2D/ReadVariableOpReadVariableOp4sequential_9_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
sequential_9/conv2d_5/Conv2DConv2Dinputs3sequential_9/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
1sequential_9/batch_normalization_9/ReadVariableOpReadVariableOp:sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:@*
dtype0?
3sequential_9/batch_normalization_9/ReadVariableOp_1ReadVariableOp<sequential_9_batch_normalization_9_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
3sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3%sequential_9/conv2d_5/Conv2D:output:09sequential_9/batch_normalization_9/ReadVariableOp:value:0;sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
is_training( ?
$sequential_9/leaky_re_lu_5/LeakyRelu	LeakyRelu7sequential_9/batch_normalization_9/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@@@?
IdentityIdentity2sequential_9/leaky_re_lu_5/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@@@?
NoOpNoOpC^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1,^sequential_9/conv2d_5/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????: : : : : 2?
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2?
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12f
1sequential_9/batch_normalization_9/ReadVariableOp1sequential_9/batch_normalization_9/ReadVariableOp2j
3sequential_9/batch_normalization_9/ReadVariableOp_13sequential_9/batch_normalization_9/ReadVariableOp_12Z
+sequential_9/conv2d_5/Conv2D/ReadVariableOp+sequential_9/conv2d_5/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
*__inference_conv2d_10_layer_call_fn_970232

inputs!
unknown:d
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_969041w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????d: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_966994

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_967058

inputs)
conv2d_5_967020:@*
batch_normalization_9_967041:@*
batch_normalization_9_967043:@*
batch_normalization_9_967045:@*
batch_normalization_9_967047:@
identity??-batch_normalization_9/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_967020*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_967019?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_9_967041batch_normalization_9_967043batch_normalization_9_967045batch_normalization_9_967047*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_967040?
leaky_re_lu_5/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_967055}
IdentityIdentity&leaky_re_lu_5/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@?
NoOpNoOp.^batch_normalization_9/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????: : : : : 2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
H__inference_conv_block_6_layer_call_and_return_conditional_losses_969981

inputsP
5sequential_10_conv2d_6_conv2d_readvariableop_resource:@?K
<sequential_10_batch_normalization_10_readvariableop_resource:	?M
>sequential_10_batch_normalization_10_readvariableop_1_resource:	?\
Msequential_10_batch_normalization_10_fusedbatchnormv3_readvariableop_resource:	?^
Osequential_10_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:	?
identity??Dsequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp?Fsequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?3sequential_10/batch_normalization_10/ReadVariableOp?5sequential_10/batch_normalization_10/ReadVariableOp_1?,sequential_10/conv2d_6/Conv2D/ReadVariableOp?
,sequential_10/conv2d_6/Conv2D/ReadVariableOpReadVariableOp5sequential_10_conv2d_6_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
sequential_10/conv2d_6/Conv2DConv2Dinputs4sequential_10/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
3sequential_10/batch_normalization_10/ReadVariableOpReadVariableOp<sequential_10_batch_normalization_10_readvariableop_resource*
_output_shapes	
:?*
dtype0?
5sequential_10/batch_normalization_10/ReadVariableOp_1ReadVariableOp>sequential_10_batch_normalization_10_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Dsequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_10_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Fsequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_10_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
5sequential_10/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3&sequential_10/conv2d_6/Conv2D:output:0;sequential_10/batch_normalization_10/ReadVariableOp:value:0=sequential_10/batch_normalization_10/ReadVariableOp_1:value:0Lsequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
is_training( ?
%sequential_10/leaky_re_lu_6/LeakyRelu	LeakyRelu9sequential_10/batch_normalization_10/FusedBatchNormV3:y:0*0
_output_shapes
:?????????  ??
IdentityIdentity3sequential_10/leaky_re_lu_6/LeakyRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????  ??
NoOpNoOpE^sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOpG^sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_14^sequential_10/batch_normalization_10/ReadVariableOp6^sequential_10/batch_normalization_10/ReadVariableOp_1-^sequential_10/conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@@@: : : : : 2?
Dsequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOpDsequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp2?
Fsequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1Fsequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12j
3sequential_10/batch_normalization_10/ReadVariableOp3sequential_10/batch_normalization_10/ReadVariableOp2n
5sequential_10/batch_normalization_10/ReadVariableOp_15sequential_10/batch_normalization_10/ReadVariableOp_12\
,sequential_10/conv2d_6/Conv2D/ReadVariableOp,sequential_10/conv2d_6/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
.__inference_discriminator_layer_call_fn_969657

inputs!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@$
	unknown_4:@?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?&

unknown_14:??

unknown_15:	?

unknown_16:	?

unknown_17:	?

unknown_18:	?%

unknown_19:?d

unknown_20:d

unknown_21:d

unknown_22:d

unknown_23:d$

unknown_24:d
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:?????????:?????????d*2
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_discriminator_layer_call_and_return_conditional_losses_969236w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????y

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
-__inference_conv_block_5_layer_call_fn_967257
input_1!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_5_layer_call_and_return_conditional_losses_967244w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
7__inference_batch_normalization_11_layer_call_fn_970959

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_967852x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_9_layer_call_fn_970270

inputs!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_967163w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
-__inference_sequential_9_layer_call_fn_970255

inputs!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_967058w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_968273

inputs
identityP
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:??????????h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
.__inference_sequential_11_layer_call_fn_968003
conv2d_7_input#
unknown:??
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_7_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_11_layer_call_and_return_conditional_losses_967975x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????  ?: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
0
_output_shapes
:?????????  ?
(
_user_specified_nameconv2d_7_input
?
?
-__inference_conv_block_5_layer_call_fn_969870

inputs!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_5_layer_call_and_return_conditional_losses_967244w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
)__inference_conv2d_9_layer_call_fn_971209

inputs"
unknown:?d
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_968643w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_6_layer_call_and_return_conditional_losses_970772

inputs9
conv2d_readvariableop_resource:@?
identity??Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????@@@: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
D__inference_conv2d_7_layer_call_and_return_conditional_losses_967831

inputs:
conv2d_readvariableop_resource:??
identity??Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:??????????^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????  ?: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_9_layer_call_fn_970637

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_966963?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
-__inference_conv_block_8_layer_call_fn_970092

inputs#
unknown:??
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_8_layer_call_and_return_conditional_losses_968462x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_10_layer_call_fn_970344

inputs"
unknown:@?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_967569x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@@@: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_9_layer_call_fn_970676

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_967110w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?

?
H__inference_conv_block_5_layer_call_and_return_conditional_losses_967289

inputs-
sequential_9_967277:@!
sequential_9_967279:@!
sequential_9_967281:@!
sequential_9_967283:@!
sequential_9_967285:@
identity??$sequential_9/StatefulPartitionedCall?
$sequential_9/StatefulPartitionedCallStatefulPartitionedCallinputssequential_9_967277sequential_9_967279sequential_9_967281sequential_9_967283sequential_9_967285*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_967163?
IdentityIdentity-sequential_9/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@m
NoOpNoOp%^sequential_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????: : : : : 2L
$sequential_9/StatefulPartitionedCall$sequential_9/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_967806

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_13_layer_call_fn_971229

inputs
unknown:d
	unknown_0:d
	unknown_1:d
	unknown_2:d
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????d*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_968587?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????d: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????d
 
_user_specified_nameinputs
?
?
)__inference_conv2d_5_layer_call_fn_970617

inputs!
unknown:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_967019w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_971008

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_9_layer_call_and_return_conditional_losses_971216

inputs9
conv2d_readvariableop_resource:?d
identity??Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?d*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d*
paddingSAME*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:?????????d^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
H__inference_conv_block_6_layer_call_and_return_conditional_losses_967738
input_1/
sequential_10_967726:@?#
sequential_10_967728:	?#
sequential_10_967730:	?#
sequential_10_967732:	?#
sequential_10_967734:	?
identity??%sequential_10/StatefulPartitionedCall?
%sequential_10/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_10_967726sequential_10_967728sequential_10_967730sequential_10_967732sequential_10_967734*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_967464?
IdentityIdentity.sequential_10/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?n
NoOpNoOp&^sequential_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@@@: : : : : 2N
%sequential_10/StatefulPartitionedCall%sequential_10/StatefulPartitionedCall:X T
/
_output_shapes
:?????????@@@
!
_user_specified_name	input_1
?
?
-__inference_conv_block_8_layer_call_fn_968475
input_1#
unknown:??
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_8_layer_call_and_return_conditional_losses_968462x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_970292

inputsA
'conv2d_5_conv2d_readvariableop_resource:@;
-batch_normalization_9_readvariableop_resource:@=
/batch_normalization_9_readvariableop_1_resource:@L
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:@
identity??5batch_normalization_9/FusedBatchNormV3/ReadVariableOp?7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_9/ReadVariableOp?&batch_normalization_9/ReadVariableOp_1?conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
:@*
dtype0?
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_5/Conv2D:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
is_training( ?
leaky_re_lu_5/LeakyRelu	LeakyRelu*batch_normalization_9/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@@@|
IdentityIdentity%leaky_re_lu_5/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@@@?
NoOpNoOp6^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1^conv2d_5/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????: : : : : 2n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_968618

inputs%
readvariableop_resource:d'
readvariableop_1_resource:d6
(fusedbatchnormv3_readvariableop_resource:d8
*fusedbatchnormv3_readvariableop_1_resource:d
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:d*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:d*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:d*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:d*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????d:d:d:d:d:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????d?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????d: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????d
 
_user_specified_nameinputs
?	
?
.__inference_sequential_12_layer_call_fn_968289
conv2d_8_input#
unknown:??
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_968276x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_8_input
?
?
7__inference_batch_normalization_13_layer_call_fn_971268

inputs
unknown:d
	unknown_0:d
	unknown_1:d
	unknown_2:d
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_968734w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????d: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_9_layer_call_fn_971345

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_968679h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????d:W S
/
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_12_layer_call_fn_971120

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_968328x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_discriminator_layer_call_fn_969352
input_1!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@$
	unknown_4:@?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?&

unknown_14:??

unknown_15:	?

unknown_16:	?

unknown_17:	?

unknown_18:	?%

unknown_19:?d

unknown_20:d

unknown_21:d

unknown_22:d

unknown_23:d$

unknown_24:d
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:?????????:?????????d*2
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_discriminator_layer_call_and_return_conditional_losses_969236w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????y

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?'
?
H__inference_conv_block_5_layer_call_and_return_conditional_losses_969929

inputsN
4sequential_9_conv2d_5_conv2d_readvariableop_resource:@H
:sequential_9_batch_normalization_9_readvariableop_resource:@J
<sequential_9_batch_normalization_9_readvariableop_1_resource:@Y
Ksequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:@[
Msequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:@
identity??1sequential_9/batch_normalization_9/AssignNewValue?3sequential_9/batch_normalization_9/AssignNewValue_1?Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp?Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?1sequential_9/batch_normalization_9/ReadVariableOp?3sequential_9/batch_normalization_9/ReadVariableOp_1?+sequential_9/conv2d_5/Conv2D/ReadVariableOp?
+sequential_9/conv2d_5/Conv2D/ReadVariableOpReadVariableOp4sequential_9_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
sequential_9/conv2d_5/Conv2DConv2Dinputs3sequential_9/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
1sequential_9/batch_normalization_9/ReadVariableOpReadVariableOp:sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:@*
dtype0?
3sequential_9/batch_normalization_9/ReadVariableOp_1ReadVariableOp<sequential_9_batch_normalization_9_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
3sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3%sequential_9/conv2d_5/Conv2D:output:09sequential_9/batch_normalization_9/ReadVariableOp:value:0;sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Jsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
1sequential_9/batch_normalization_9/AssignNewValueAssignVariableOpKsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource@sequential_9/batch_normalization_9/FusedBatchNormV3:batch_mean:0C^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
3sequential_9/batch_normalization_9/AssignNewValue_1AssignVariableOpMsequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resourceDsequential_9/batch_normalization_9/FusedBatchNormV3:batch_variance:0E^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
$sequential_9/leaky_re_lu_5/LeakyRelu	LeakyRelu7sequential_9/batch_normalization_9/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@@@?
IdentityIdentity2sequential_9/leaky_re_lu_5/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@@@?
NoOpNoOp2^sequential_9/batch_normalization_9/AssignNewValue4^sequential_9/batch_normalization_9/AssignNewValue_1C^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpE^sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12^sequential_9/batch_normalization_9/ReadVariableOp4^sequential_9/batch_normalization_9/ReadVariableOp_1,^sequential_9/conv2d_5/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????: : : : : 2f
1sequential_9/batch_normalization_9/AssignNewValue1sequential_9/batch_normalization_9/AssignNewValue2j
3sequential_9/batch_normalization_9/AssignNewValue_13sequential_9/batch_normalization_9/AssignNewValue_12?
Bsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpBsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2?
Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Dsequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12f
1sequential_9/batch_normalization_9/ReadVariableOp1sequential_9/batch_normalization_9/ReadVariableOp2j
3sequential_9/batch_normalization_9/ReadVariableOp_13sequential_9/batch_normalization_9/ReadVariableOp_12Z
+sequential_9/conv2d_5/Conv2D/ReadVariableOp+sequential_9/conv2d_5/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_967775

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
.__inference_sequential_10_layer_call_fn_967477
conv2d_6_input"
unknown:@?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_967464x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@@@: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????@@@
(
_user_specified_nameconv2d_6_input
?
?
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_970712

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_10_layer_call_fn_970811

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_967446x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????  ?: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?+
?

I__inference_discriminator_layer_call_and_return_conditional_losses_969236

inputs-
conv_block_5_969176:@!
conv_block_5_969178:@!
conv_block_5_969180:@!
conv_block_5_969182:@!
conv_block_5_969184:@.
conv_block_6_969187:@?"
conv_block_6_969189:	?"
conv_block_6_969191:	?"
conv_block_6_969193:	?"
conv_block_6_969195:	?/
conv_block_7_969198:??"
conv_block_7_969200:	?"
conv_block_7_969202:	?"
conv_block_7_969204:	?"
conv_block_7_969206:	?/
conv_block_8_969209:??"
conv_block_8_969211:	?"
conv_block_8_969213:	?"
conv_block_8_969215:	?"
conv_block_8_969217:	?.
conv_block_9_969220:?d!
conv_block_9_969222:d!
conv_block_9_969224:d!
conv_block_9_969226:d!
conv_block_9_969228:d*
conv2d_10_969231:d
identity

identity_1??!conv2d_10/StatefulPartitionedCall?$conv_block_5/StatefulPartitionedCall?$conv_block_6/StatefulPartitionedCall?$conv_block_7/StatefulPartitionedCall?$conv_block_8/StatefulPartitionedCall?$conv_block_9/StatefulPartitionedCall?
$conv_block_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv_block_5_969176conv_block_5_969178conv_block_5_969180conv_block_5_969182conv_block_5_969184*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_5_layer_call_and_return_conditional_losses_967289?
$conv_block_6/StatefulPartitionedCallStatefulPartitionedCall-conv_block_5/StatefulPartitionedCall:output:0conv_block_6_969187conv_block_6_969189conv_block_6_969191conv_block_6_969193conv_block_6_969195*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_6_layer_call_and_return_conditional_losses_967695?
$conv_block_7/StatefulPartitionedCallStatefulPartitionedCall-conv_block_6/StatefulPartitionedCall:output:0conv_block_7_969198conv_block_7_969200conv_block_7_969202conv_block_7_969204conv_block_7_969206*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_7_layer_call_and_return_conditional_losses_968101?
$conv_block_8/StatefulPartitionedCallStatefulPartitionedCall-conv_block_7/StatefulPartitionedCall:output:0conv_block_8_969209conv_block_8_969211conv_block_8_969213conv_block_8_969215conv_block_8_969217*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_8_layer_call_and_return_conditional_losses_968507?
$conv_block_9/StatefulPartitionedCallStatefulPartitionedCall-conv_block_8/StatefulPartitionedCall:output:0conv_block_9_969220conv_block_9_969222conv_block_9_969224conv_block_9_969226conv_block_9_969228*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_9_layer_call_and_return_conditional_losses_968913?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall-conv_block_9/StatefulPartitionedCall:output:0conv2d_10_969231*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_969041?
IdentityIdentity*conv2d_10/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????

Identity_1Identity-conv_block_9/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????d?
NoOpNoOp"^conv2d_10/StatefulPartitionedCall%^conv_block_5/StatefulPartitionedCall%^conv_block_6/StatefulPartitionedCall%^conv_block_7/StatefulPartitionedCall%^conv_block_8/StatefulPartitionedCall%^conv_block_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2L
$conv_block_5/StatefulPartitionedCall$conv_block_5/StatefulPartitionedCall2L
$conv_block_6/StatefulPartitionedCall$conv_block_6/StatefulPartitionedCall2L
$conv_block_7/StatefulPartitionedCall$conv_block_7/StatefulPartitionedCall2L
$conv_block_8/StatefulPartitionedCall$conv_block_8/StatefulPartitionedCall2L
$conv_block_9/StatefulPartitionedCall$conv_block_9/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_10_layer_call_fn_970798

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_967400?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?

?
H__inference_conv_block_6_layer_call_and_return_conditional_losses_967695

inputs/
sequential_10_967683:@?#
sequential_10_967685:	?#
sequential_10_967687:	?#
sequential_10_967689:	?#
sequential_10_967691:	?
identity??%sequential_10/StatefulPartitionedCall?
%sequential_10/StatefulPartitionedCallStatefulPartitionedCallinputssequential_10_967683sequential_10_967685sequential_10_967687sequential_10_967689sequential_10_967691*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_967569?
IdentityIdentity.sequential_10/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?n
NoOpNoOp&^sequential_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@@@: : : : : 2N
%sequential_10/StatefulPartitionedCall%sequential_10/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_970906

inputs
identityP
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:?????????  ?h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:?????????  ?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?+
?

I__inference_discriminator_layer_call_and_return_conditional_losses_969478
input_1-
conv_block_5_969418:@!
conv_block_5_969420:@!
conv_block_5_969422:@!
conv_block_5_969424:@!
conv_block_5_969426:@.
conv_block_6_969429:@?"
conv_block_6_969431:	?"
conv_block_6_969433:	?"
conv_block_6_969435:	?"
conv_block_6_969437:	?/
conv_block_7_969440:??"
conv_block_7_969442:	?"
conv_block_7_969444:	?"
conv_block_7_969446:	?"
conv_block_7_969448:	?/
conv_block_8_969451:??"
conv_block_8_969453:	?"
conv_block_8_969455:	?"
conv_block_8_969457:	?"
conv_block_8_969459:	?.
conv_block_9_969462:?d!
conv_block_9_969464:d!
conv_block_9_969466:d!
conv_block_9_969468:d!
conv_block_9_969470:d*
conv2d_10_969473:d
identity

identity_1??!conv2d_10/StatefulPartitionedCall?$conv_block_5/StatefulPartitionedCall?$conv_block_6/StatefulPartitionedCall?$conv_block_7/StatefulPartitionedCall?$conv_block_8/StatefulPartitionedCall?$conv_block_9/StatefulPartitionedCall?
$conv_block_5/StatefulPartitionedCallStatefulPartitionedCallinput_1conv_block_5_969418conv_block_5_969420conv_block_5_969422conv_block_5_969424conv_block_5_969426*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_5_layer_call_and_return_conditional_losses_967289?
$conv_block_6/StatefulPartitionedCallStatefulPartitionedCall-conv_block_5/StatefulPartitionedCall:output:0conv_block_6_969429conv_block_6_969431conv_block_6_969433conv_block_6_969435conv_block_6_969437*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_6_layer_call_and_return_conditional_losses_967695?
$conv_block_7/StatefulPartitionedCallStatefulPartitionedCall-conv_block_6/StatefulPartitionedCall:output:0conv_block_7_969440conv_block_7_969442conv_block_7_969444conv_block_7_969446conv_block_7_969448*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_7_layer_call_and_return_conditional_losses_968101?
$conv_block_8/StatefulPartitionedCallStatefulPartitionedCall-conv_block_7/StatefulPartitionedCall:output:0conv_block_8_969451conv_block_8_969453conv_block_8_969455conv_block_8_969457conv_block_8_969459*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_8_layer_call_and_return_conditional_losses_968507?
$conv_block_9/StatefulPartitionedCallStatefulPartitionedCall-conv_block_8/StatefulPartitionedCall:output:0conv_block_9_969462conv_block_9_969464conv_block_9_969466conv_block_9_969468conv_block_9_969470*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_9_layer_call_and_return_conditional_losses_968913?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall-conv_block_9/StatefulPartitionedCall:output:0conv2d_10_969473*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_969041?
IdentityIdentity*conv2d_10/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????

Identity_1Identity-conv_block_9/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????d?
NoOpNoOp"^conv2d_10/StatefulPartitionedCall%^conv_block_5/StatefulPartitionedCall%^conv_block_6/StatefulPartitionedCall%^conv_block_7/StatefulPartitionedCall%^conv_block_8/StatefulPartitionedCall%^conv_block_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2L
$conv_block_5/StatefulPartitionedCall$conv_block_5/StatefulPartitionedCall2L
$conv_block_6/StatefulPartitionedCall$conv_block_6/StatefulPartitionedCall2L
$conv_block_7/StatefulPartitionedCall$conv_block_7/StatefulPartitionedCall2L
$conv_block_8/StatefulPartitionedCall$conv_block_8/StatefulPartitionedCall2L
$conv_block_9/StatefulPartitionedCall$conv_block_9/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?	
?
7__inference_batch_normalization_12_layer_call_fn_971094

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_968212?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_971026

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
.__inference_sequential_13_layer_call_fn_968695
conv2d_9_input"
unknown:?d
	unknown_0:d
	unknown_1:d
	unknown_2:d
	unknown_3:d
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_13_layer_call_and_return_conditional_losses_968682w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_9_input
?
?
I__inference_sequential_11_layer_call_and_return_conditional_losses_968037
conv2d_7_input+
conv2d_7_968023:??,
batch_normalization_11_968026:	?,
batch_normalization_11_968028:	?,
batch_normalization_11_968030:	?,
batch_normalization_11_968032:	?
identity??.batch_normalization_11/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCallconv2d_7_inputconv2d_7_968023*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_967831?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_11_968026batch_normalization_11_968028batch_normalization_11_968030batch_normalization_11_968032*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_967922?
leaky_re_lu_7/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_967867~
IdentityIdentity&leaky_re_lu_7/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp/^batch_normalization_11/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????  ?: : : : : 2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:` \
0
_output_shapes
:?????????  ?
(
_user_specified_nameconv2d_7_input
?
?
.__inference_sequential_12_layer_call_fn_970477

inputs#
unknown:??
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_968276x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
-__inference_conv_block_9_layer_call_fn_968881
input_1"
unknown:?d
	unknown_0:d
	unknown_1:d
	unknown_2:d
	unknown_3:d
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_9_layer_call_and_return_conditional_losses_968868w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
D__inference_conv2d_7_layer_call_and_return_conditional_losses_970920

inputs:
conv2d_readvariableop_resource:??
identity??Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:??????????^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????  ?: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
I__inference_sequential_13_layer_call_and_return_conditional_losses_968832
conv2d_9_input*
conv2d_9_968818:?d+
batch_normalization_13_968821:d+
batch_normalization_13_968823:d+
batch_normalization_13_968825:d+
batch_normalization_13_968827:d
identity??.batch_normalization_13/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallconv2d_9_inputconv2d_9_968818*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_968643?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0batch_normalization_13_968821batch_normalization_13_968823batch_normalization_13_968825batch_normalization_13_968827*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_968664?
leaky_re_lu_9/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_968679}
IdentityIdentity&leaky_re_lu_9/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????d?
NoOpNoOp/^batch_normalization_13/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:` \
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_9_input
?	
?
7__inference_batch_normalization_10_layer_call_fn_970785

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_967369?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
-__inference_conv_block_8_layer_call_fn_968535
input_1#
unknown:??
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_8_layer_call_and_return_conditional_losses_968507x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_967040

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
D__inference_conv2d_8_layer_call_and_return_conditional_losses_968237

inputs:
conv2d_readvariableop_resource:??
identity??Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:??????????^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?i
?
"__inference__traced_restore_971540
file_prefixI
/assignvariableop_discriminator_conv2d_10_kernel:d<
"assignvariableop_1_conv2d_5_kernel:@<
.assignvariableop_2_batch_normalization_9_gamma:@;
-assignvariableop_3_batch_normalization_9_beta:@B
4assignvariableop_4_batch_normalization_9_moving_mean:@F
8assignvariableop_5_batch_normalization_9_moving_variance:@=
"assignvariableop_6_conv2d_6_kernel:@?>
/assignvariableop_7_batch_normalization_10_gamma:	?=
.assignvariableop_8_batch_normalization_10_beta:	?D
5assignvariableop_9_batch_normalization_10_moving_mean:	?I
:assignvariableop_10_batch_normalization_10_moving_variance:	??
#assignvariableop_11_conv2d_7_kernel:???
0assignvariableop_12_batch_normalization_11_gamma:	?>
/assignvariableop_13_batch_normalization_11_beta:	?E
6assignvariableop_14_batch_normalization_11_moving_mean:	?I
:assignvariableop_15_batch_normalization_11_moving_variance:	??
#assignvariableop_16_conv2d_8_kernel:???
0assignvariableop_17_batch_normalization_12_gamma:	?>
/assignvariableop_18_batch_normalization_12_beta:	?E
6assignvariableop_19_batch_normalization_12_moving_mean:	?I
:assignvariableop_20_batch_normalization_12_moving_variance:	?>
#assignvariableop_21_conv2d_9_kernel:?d>
0assignvariableop_22_batch_normalization_13_gamma:d=
/assignvariableop_23_batch_normalization_13_beta:dD
6assignvariableop_24_batch_normalization_13_moving_mean:dH
:assignvariableop_25_batch_normalization_13_moving_variance:d
identity_27??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B)outputs/kernel/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp/assignvariableop_discriminator_conv2d_10_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv2d_5_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_9_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp-assignvariableop_3_batch_normalization_9_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp4assignvariableop_4_batch_normalization_9_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp8assignvariableop_5_batch_normalization_9_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_6_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp/assignvariableop_7_batch_normalization_10_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_10_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp5assignvariableop_9_batch_normalization_10_moving_meanIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp:assignvariableop_10_batch_normalization_10_moving_varianceIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp#assignvariableop_11_conv2d_7_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp0assignvariableop_12_batch_normalization_11_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp/assignvariableop_13_batch_normalization_11_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp6assignvariableop_14_batch_normalization_11_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp:assignvariableop_15_batch_normalization_11_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp#assignvariableop_16_conv2d_8_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp0assignvariableop_17_batch_normalization_12_gammaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp/assignvariableop_18_batch_normalization_12_betaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp6assignvariableop_19_batch_normalization_12_moving_meanIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp:assignvariableop_20_batch_normalization_12_moving_varianceIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp#assignvariableop_21_conv2d_9_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp0assignvariableop_22_batch_normalization_13_gammaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp/assignvariableop_23_batch_normalization_13_betaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp6assignvariableop_24_batch_normalization_13_moving_meanIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp:assignvariableop_25_batch_normalization_13_moving_varianceIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_26Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_27IdentityIdentity_26:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_27Identity_27:output:0*I
_input_shapes8
6: : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_25AssignVariableOp_252(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?(
?
H__inference_conv_block_6_layer_call_and_return_conditional_losses_970003

inputsP
5sequential_10_conv2d_6_conv2d_readvariableop_resource:@?K
<sequential_10_batch_normalization_10_readvariableop_resource:	?M
>sequential_10_batch_normalization_10_readvariableop_1_resource:	?\
Msequential_10_batch_normalization_10_fusedbatchnormv3_readvariableop_resource:	?^
Osequential_10_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:	?
identity??3sequential_10/batch_normalization_10/AssignNewValue?5sequential_10/batch_normalization_10/AssignNewValue_1?Dsequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp?Fsequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?3sequential_10/batch_normalization_10/ReadVariableOp?5sequential_10/batch_normalization_10/ReadVariableOp_1?,sequential_10/conv2d_6/Conv2D/ReadVariableOp?
,sequential_10/conv2d_6/Conv2D/ReadVariableOpReadVariableOp5sequential_10_conv2d_6_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
sequential_10/conv2d_6/Conv2DConv2Dinputs4sequential_10/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
3sequential_10/batch_normalization_10/ReadVariableOpReadVariableOp<sequential_10_batch_normalization_10_readvariableop_resource*
_output_shapes	
:?*
dtype0?
5sequential_10/batch_normalization_10/ReadVariableOp_1ReadVariableOp>sequential_10_batch_normalization_10_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Dsequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_10_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Fsequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_10_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
5sequential_10/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3&sequential_10/conv2d_6/Conv2D:output:0;sequential_10/batch_normalization_10/ReadVariableOp:value:0=sequential_10/batch_normalization_10/ReadVariableOp_1:value:0Lsequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
3sequential_10/batch_normalization_10/AssignNewValueAssignVariableOpMsequential_10_batch_normalization_10_fusedbatchnormv3_readvariableop_resourceBsequential_10/batch_normalization_10/FusedBatchNormV3:batch_mean:0E^sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
5sequential_10/batch_normalization_10/AssignNewValue_1AssignVariableOpOsequential_10_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resourceFsequential_10/batch_normalization_10/FusedBatchNormV3:batch_variance:0G^sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
%sequential_10/leaky_re_lu_6/LeakyRelu	LeakyRelu9sequential_10/batch_normalization_10/FusedBatchNormV3:y:0*0
_output_shapes
:?????????  ??
IdentityIdentity3sequential_10/leaky_re_lu_6/LeakyRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????  ??
NoOpNoOp4^sequential_10/batch_normalization_10/AssignNewValue6^sequential_10/batch_normalization_10/AssignNewValue_1E^sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOpG^sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_14^sequential_10/batch_normalization_10/ReadVariableOp6^sequential_10/batch_normalization_10/ReadVariableOp_1-^sequential_10/conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@@@: : : : : 2j
3sequential_10/batch_normalization_10/AssignNewValue3sequential_10/batch_normalization_10/AssignNewValue2n
5sequential_10/batch_normalization_10/AssignNewValue_15sequential_10/batch_normalization_10/AssignNewValue_12?
Dsequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOpDsequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp2?
Fsequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1Fsequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12j
3sequential_10/batch_normalization_10/ReadVariableOp3sequential_10/batch_normalization_10/ReadVariableOp2n
5sequential_10/batch_normalization_10/ReadVariableOp_15sequential_10/batch_normalization_10/ReadVariableOp_12\
,sequential_10/conv2d_6/Conv2D/ReadVariableOp,sequential_10/conv2d_6/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
-__inference_conv_block_5_layer_call_fn_967317
input_1!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_5_layer_call_and_return_conditional_losses_967289w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_968734

inputs%
readvariableop_resource:d'
readvariableop_1_resource:d6
(fusedbatchnormv3_readvariableop_resource:d8
*fusedbatchnormv3_readvariableop_1_resource:d
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:d*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:d*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:d*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:d*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????d:d:d:d:d:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????d?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????d: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_971192

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_11_layer_call_fn_970403

inputs#
unknown:??
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_11_layer_call_and_return_conditional_losses_967870x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????  ?: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
I__inference_sequential_13_layer_call_and_return_conditional_losses_970588

inputsB
'conv2d_9_conv2d_readvariableop_resource:?d<
.batch_normalization_13_readvariableop_resource:d>
0batch_normalization_13_readvariableop_1_resource:dM
?batch_normalization_13_fusedbatchnormv3_readvariableop_resource:dO
Abatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:d
identity??6batch_normalization_13/FusedBatchNormV3/ReadVariableOp?8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_13/ReadVariableOp?'batch_normalization_13/ReadVariableOp_1?conv2d_9/Conv2D/ReadVariableOp?
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*'
_output_shapes
:?d*
dtype0?
conv2d_9/Conv2DConv2Dinputs&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d*
paddingSAME*
strides
?
%batch_normalization_13/ReadVariableOpReadVariableOp.batch_normalization_13_readvariableop_resource*
_output_shapes
:d*
dtype0?
'batch_normalization_13/ReadVariableOp_1ReadVariableOp0batch_normalization_13_readvariableop_1_resource*
_output_shapes
:d*
dtype0?
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:d*
dtype0?
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:d*
dtype0?
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3conv2d_9/Conv2D:output:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????d:d:d:d:d:*
epsilon%o?:*
is_training( ?
leaky_re_lu_9/LeakyRelu	LeakyRelu+batch_normalization_13/FusedBatchNormV3:y:0*/
_output_shapes
:?????????d|
IdentityIdentity%leaky_re_lu_9/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????d?
NoOpNoOp7^batch_normalization_13/FusedBatchNormV3/ReadVariableOp9^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_13/ReadVariableOp(^batch_normalization_13/ReadVariableOp_1^conv2d_9/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 2p
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp6batch_normalization_13/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_18batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_13/ReadVariableOp%batch_normalization_13/ReadVariableOp2R
'batch_normalization_13/ReadVariableOp_1'batch_normalization_13/ReadVariableOp_12@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_7_layer_call_fn_971049

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_967867i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_968212

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
H__inference_conv_block_7_layer_call_and_return_conditional_losses_970055

inputsQ
5sequential_11_conv2d_7_conv2d_readvariableop_resource:??K
<sequential_11_batch_normalization_11_readvariableop_resource:	?M
>sequential_11_batch_normalization_11_readvariableop_1_resource:	?\
Msequential_11_batch_normalization_11_fusedbatchnormv3_readvariableop_resource:	?^
Osequential_11_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:	?
identity??Dsequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp?Fsequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1?3sequential_11/batch_normalization_11/ReadVariableOp?5sequential_11/batch_normalization_11/ReadVariableOp_1?,sequential_11/conv2d_7/Conv2D/ReadVariableOp?
,sequential_11/conv2d_7/Conv2D/ReadVariableOpReadVariableOp5sequential_11_conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
sequential_11/conv2d_7/Conv2DConv2Dinputs4sequential_11/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
3sequential_11/batch_normalization_11/ReadVariableOpReadVariableOp<sequential_11_batch_normalization_11_readvariableop_resource*
_output_shapes	
:?*
dtype0?
5sequential_11/batch_normalization_11/ReadVariableOp_1ReadVariableOp>sequential_11_batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Dsequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_11_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Fsequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_11_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
5sequential_11/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3&sequential_11/conv2d_7/Conv2D:output:0;sequential_11/batch_normalization_11/ReadVariableOp:value:0=sequential_11/batch_normalization_11/ReadVariableOp_1:value:0Lsequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( ?
%sequential_11/leaky_re_lu_7/LeakyRelu	LeakyRelu9sequential_11/batch_normalization_11/FusedBatchNormV3:y:0*0
_output_shapes
:???????????
IdentityIdentity3sequential_11/leaky_re_lu_7/LeakyRelu:activations:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOpE^sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOpG^sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_14^sequential_11/batch_normalization_11/ReadVariableOp6^sequential_11/batch_normalization_11/ReadVariableOp_1-^sequential_11/conv2d_7/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????  ?: : : : : 2?
Dsequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOpDsequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp2?
Fsequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1Fsequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12j
3sequential_11/batch_normalization_11/ReadVariableOp3sequential_11/batch_normalization_11/ReadVariableOp2n
5sequential_11/batch_normalization_11/ReadVariableOp_15sequential_11/batch_normalization_11/ReadVariableOp_12\
,sequential_11/conv2d_7/Conv2D/ReadVariableOp,sequential_11/conv2d_7/Conv2D/ReadVariableOp:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?

?
H__inference_conv_block_8_layer_call_and_return_conditional_losses_968507

inputs0
sequential_12_968495:??#
sequential_12_968497:	?#
sequential_12_968499:	?#
sequential_12_968501:	?#
sequential_12_968503:	?
identity??%sequential_12/StatefulPartitionedCall?
%sequential_12/StatefulPartitionedCallStatefulPartitionedCallinputssequential_12_968495sequential_12_968497sequential_12_968499sequential_12_968501sequential_12_968503*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_968381?
IdentityIdentity.sequential_12/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????n
NoOpNoOp&^sequential_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 2N
%sequential_12/StatefulPartitionedCall%sequential_12/StatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_971054

inputs
identityP
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:??????????h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_12_layer_call_and_return_conditional_losses_970514

inputsC
'conv2d_8_conv2d_readvariableop_resource:??=
.batch_normalization_12_readvariableop_resource:	??
0batch_normalization_12_readvariableop_1_resource:	?N
?batch_normalization_12_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:	?
identity??6batch_normalization_12/FusedBatchNormV3/ReadVariableOp?8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_12/ReadVariableOp?'batch_normalization_12/ReadVariableOp_1?conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_8/Conv2DConv2Dinputs&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
%batch_normalization_12/ReadVariableOpReadVariableOp.batch_normalization_12_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_12/ReadVariableOp_1ReadVariableOp0batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3conv2d_8/Conv2D:output:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( ?
leaky_re_lu_8/LeakyRelu	LeakyRelu+batch_normalization_12/FusedBatchNormV3:y:0*0
_output_shapes
:??????????}
IdentityIdentity%leaky_re_lu_8/LeakyRelu:activations:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp7^batch_normalization_12/FusedBatchNormV3/ReadVariableOp9^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_12/ReadVariableOp(^batch_normalization_12/ReadVariableOp_1^conv2d_8/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 2p
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp6batch_normalization_12/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_18batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_12/ReadVariableOp%batch_normalization_12/ReadVariableOp2R
'batch_normalization_12/ReadVariableOp_1'batch_normalization_12/ReadVariableOp_12@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_967516

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????  ??
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????  ?: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?

?
H__inference_conv_block_7_layer_call_and_return_conditional_losses_968159
input_10
sequential_11_968147:??#
sequential_11_968149:	?#
sequential_11_968151:	?#
sequential_11_968153:	?#
sequential_11_968155:	?
identity??%sequential_11/StatefulPartitionedCall?
%sequential_11/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_11_968147sequential_11_968149sequential_11_968151sequential_11_968153sequential_11_968155*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_11_layer_call_and_return_conditional_losses_967975?
IdentityIdentity.sequential_11/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????n
NoOpNoOp&^sequential_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????  ?: : : : : 2N
%sequential_11/StatefulPartitionedCall%sequential_11/StatefulPartitionedCall:Y U
0
_output_shapes
:?????????  ?
!
_user_specified_name	input_1
?
?
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_970990

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?(
?
H__inference_conv_block_7_layer_call_and_return_conditional_losses_970077

inputsQ
5sequential_11_conv2d_7_conv2d_readvariableop_resource:??K
<sequential_11_batch_normalization_11_readvariableop_resource:	?M
>sequential_11_batch_normalization_11_readvariableop_1_resource:	?\
Msequential_11_batch_normalization_11_fusedbatchnormv3_readvariableop_resource:	?^
Osequential_11_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:	?
identity??3sequential_11/batch_normalization_11/AssignNewValue?5sequential_11/batch_normalization_11/AssignNewValue_1?Dsequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp?Fsequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1?3sequential_11/batch_normalization_11/ReadVariableOp?5sequential_11/batch_normalization_11/ReadVariableOp_1?,sequential_11/conv2d_7/Conv2D/ReadVariableOp?
,sequential_11/conv2d_7/Conv2D/ReadVariableOpReadVariableOp5sequential_11_conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
sequential_11/conv2d_7/Conv2DConv2Dinputs4sequential_11/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
3sequential_11/batch_normalization_11/ReadVariableOpReadVariableOp<sequential_11_batch_normalization_11_readvariableop_resource*
_output_shapes	
:?*
dtype0?
5sequential_11/batch_normalization_11/ReadVariableOp_1ReadVariableOp>sequential_11_batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Dsequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_11_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Fsequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_11_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
5sequential_11/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3&sequential_11/conv2d_7/Conv2D:output:0;sequential_11/batch_normalization_11/ReadVariableOp:value:0=sequential_11/batch_normalization_11/ReadVariableOp_1:value:0Lsequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
3sequential_11/batch_normalization_11/AssignNewValueAssignVariableOpMsequential_11_batch_normalization_11_fusedbatchnormv3_readvariableop_resourceBsequential_11/batch_normalization_11/FusedBatchNormV3:batch_mean:0E^sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
5sequential_11/batch_normalization_11/AssignNewValue_1AssignVariableOpOsequential_11_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resourceFsequential_11/batch_normalization_11/FusedBatchNormV3:batch_variance:0G^sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
%sequential_11/leaky_re_lu_7/LeakyRelu	LeakyRelu9sequential_11/batch_normalization_11/FusedBatchNormV3:y:0*0
_output_shapes
:???????????
IdentityIdentity3sequential_11/leaky_re_lu_7/LeakyRelu:activations:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp4^sequential_11/batch_normalization_11/AssignNewValue6^sequential_11/batch_normalization_11/AssignNewValue_1E^sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOpG^sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_14^sequential_11/batch_normalization_11/ReadVariableOp6^sequential_11/batch_normalization_11/ReadVariableOp_1-^sequential_11/conv2d_7/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????  ?: : : : : 2j
3sequential_11/batch_normalization_11/AssignNewValue3sequential_11/batch_normalization_11/AssignNewValue2n
5sequential_11/batch_normalization_11/AssignNewValue_15sequential_11/batch_normalization_11/AssignNewValue_12?
Dsequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOpDsequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp2?
Fsequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1Fsequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12j
3sequential_11/batch_normalization_11/ReadVariableOp3sequential_11/batch_normalization_11/ReadVariableOp2n
5sequential_11/batch_normalization_11/ReadVariableOp_15sequential_11/batch_normalization_11/ReadVariableOp_12\
,sequential_11/conv2d_7/Conv2D/ReadVariableOp,sequential_11/conv2d_7/Conv2D/ReadVariableOp:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?+
?

I__inference_discriminator_layer_call_and_return_conditional_losses_969415
input_1-
conv_block_5_969355:@!
conv_block_5_969357:@!
conv_block_5_969359:@!
conv_block_5_969361:@!
conv_block_5_969363:@.
conv_block_6_969366:@?"
conv_block_6_969368:	?"
conv_block_6_969370:	?"
conv_block_6_969372:	?"
conv_block_6_969374:	?/
conv_block_7_969377:??"
conv_block_7_969379:	?"
conv_block_7_969381:	?"
conv_block_7_969383:	?"
conv_block_7_969385:	?/
conv_block_8_969388:??"
conv_block_8_969390:	?"
conv_block_8_969392:	?"
conv_block_8_969394:	?"
conv_block_8_969396:	?.
conv_block_9_969399:?d!
conv_block_9_969401:d!
conv_block_9_969403:d!
conv_block_9_969405:d!
conv_block_9_969407:d*
conv2d_10_969410:d
identity

identity_1??!conv2d_10/StatefulPartitionedCall?$conv_block_5/StatefulPartitionedCall?$conv_block_6/StatefulPartitionedCall?$conv_block_7/StatefulPartitionedCall?$conv_block_8/StatefulPartitionedCall?$conv_block_9/StatefulPartitionedCall?
$conv_block_5/StatefulPartitionedCallStatefulPartitionedCallinput_1conv_block_5_969355conv_block_5_969357conv_block_5_969359conv_block_5_969361conv_block_5_969363*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_5_layer_call_and_return_conditional_losses_967244?
$conv_block_6/StatefulPartitionedCallStatefulPartitionedCall-conv_block_5/StatefulPartitionedCall:output:0conv_block_6_969366conv_block_6_969368conv_block_6_969370conv_block_6_969372conv_block_6_969374*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_6_layer_call_and_return_conditional_losses_967650?
$conv_block_7/StatefulPartitionedCallStatefulPartitionedCall-conv_block_6/StatefulPartitionedCall:output:0conv_block_7_969377conv_block_7_969379conv_block_7_969381conv_block_7_969383conv_block_7_969385*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_7_layer_call_and_return_conditional_losses_968056?
$conv_block_8/StatefulPartitionedCallStatefulPartitionedCall-conv_block_7/StatefulPartitionedCall:output:0conv_block_8_969388conv_block_8_969390conv_block_8_969392conv_block_8_969394conv_block_8_969396*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_8_layer_call_and_return_conditional_losses_968462?
$conv_block_9/StatefulPartitionedCallStatefulPartitionedCall-conv_block_8/StatefulPartitionedCall:output:0conv_block_9_969399conv_block_9_969401conv_block_9_969403conv_block_9_969405conv_block_9_969407*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_9_layer_call_and_return_conditional_losses_968868?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall-conv_block_9/StatefulPartitionedCall:output:0conv2d_10_969410*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_969041?
IdentityIdentity*conv2d_10/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????

Identity_1Identity-conv_block_9/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????d?
NoOpNoOp"^conv2d_10/StatefulPartitionedCall%^conv_block_5/StatefulPartitionedCall%^conv_block_6/StatefulPartitionedCall%^conv_block_7/StatefulPartitionedCall%^conv_block_8/StatefulPartitionedCall%^conv_block_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2L
$conv_block_5/StatefulPartitionedCall$conv_block_5/StatefulPartitionedCall2L
$conv_block_6/StatefulPartitionedCall$conv_block_6/StatefulPartitionedCall2L
$conv_block_7/StatefulPartitionedCall$conv_block_7/StatefulPartitionedCall2L
$conv_block_8/StatefulPartitionedCall$conv_block_8/StatefulPartitionedCall2L
$conv_block_9/StatefulPartitionedCall$conv_block_9/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
e
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_970758

inputs
identityO
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????@@@g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@@:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
D__inference_conv2d_5_layer_call_and_return_conditional_losses_970624

inputs8
conv2d_readvariableop_resource:@
identity??Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
-__inference_conv_block_9_layer_call_fn_968941
input_1"
unknown:?d
	unknown_0:d
	unknown_1:d
	unknown_2:d
	unknown_3:d
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_9_layer_call_and_return_conditional_losses_968913w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
.__inference_sequential_13_layer_call_fn_970551

inputs"
unknown:?d
	unknown_0:d
	unknown_1:d
	unknown_2:d
	unknown_3:d
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_13_layer_call_and_return_conditional_losses_968682w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_8_layer_call_fn_971197

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_968273i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_11_layer_call_fn_970418

inputs#
unknown:??
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_11_layer_call_and_return_conditional_losses_967975x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????  ?: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?

?
H__inference_conv_block_8_layer_call_and_return_conditional_losses_968565
input_10
sequential_12_968553:??#
sequential_12_968555:	?#
sequential_12_968557:	?#
sequential_12_968559:	?#
sequential_12_968561:	?
identity??%sequential_12/StatefulPartitionedCall?
%sequential_12/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_12_968553sequential_12_968555sequential_12_968557sequential_12_968559sequential_12_968561*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_968381?
IdentityIdentity.sequential_12/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????n
NoOpNoOp&^sequential_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 2N
%sequential_12/StatefulPartitionedCall%sequential_12/StatefulPartitionedCall:Y U
0
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
-__inference_conv_block_7_layer_call_fn_968129
input_1#
unknown:??
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_7_layer_call_and_return_conditional_losses_968101x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????  ?: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:?????????  ?
!
_user_specified_name	input_1
?
?
)__inference_conv2d_8_layer_call_fn_971061

inputs#
unknown:??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_968237x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?!
?
I__inference_sequential_11_layer_call_and_return_conditional_losses_970462

inputsC
'conv2d_7_conv2d_readvariableop_resource:??=
.batch_normalization_11_readvariableop_resource:	??
0batch_normalization_11_readvariableop_1_resource:	?N
?batch_normalization_11_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:	?
identity??%batch_normalization_11/AssignNewValue?'batch_normalization_11/AssignNewValue_1?6batch_normalization_11/FusedBatchNormV3/ReadVariableOp?8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_11/ReadVariableOp?'batch_normalization_11/ReadVariableOp_1?conv2d_7/Conv2D/ReadVariableOp?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_7/Conv2DConv2Dinputs&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_7/Conv2D:output:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_11/AssignNewValueAssignVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource4batch_normalization_11/FusedBatchNormV3:batch_mean:07^batch_normalization_11/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization_11/AssignNewValue_1AssignVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_11/FusedBatchNormV3:batch_variance:09^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_re_lu_7/LeakyRelu	LeakyRelu+batch_normalization_11/FusedBatchNormV3:y:0*0
_output_shapes
:??????????}
IdentityIdentity%leaky_re_lu_7/LeakyRelu:activations:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp&^batch_normalization_11/AssignNewValue(^batch_normalization_11/AssignNewValue_17^batch_normalization_11/FusedBatchNormV3/ReadVariableOp9^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_11/ReadVariableOp(^batch_normalization_11/ReadVariableOp_1^conv2d_7/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????  ?: : : : : 2N
%batch_normalization_11/AssignNewValue%batch_normalization_11/AssignNewValue2R
'batch_normalization_11/AssignNewValue_1'batch_normalization_11/AssignNewValue_12p
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp6batch_normalization_11/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_18batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_11/ReadVariableOp%batch_normalization_11/ReadVariableOp2R
'batch_normalization_11/ReadVariableOp_1'batch_normalization_11/ReadVariableOp_12@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
I__inference_sequential_11_layer_call_and_return_conditional_losses_968020
conv2d_7_input+
conv2d_7_968006:??,
batch_normalization_11_968009:	?,
batch_normalization_11_968011:	?,
batch_normalization_11_968013:	?,
batch_normalization_11_968015:	?
identity??.batch_normalization_11/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCallconv2d_7_inputconv2d_7_968006*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_967831?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_11_968009batch_normalization_11_968011batch_normalization_11_968013batch_normalization_11_968015*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_967852?
leaky_re_lu_7/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_967867~
IdentityIdentity&leaky_re_lu_7/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp/^batch_normalization_11/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????  ?: : : : : 2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:` \
0
_output_shapes
:?????????  ?
(
_user_specified_nameconv2d_7_input
?!
?
I__inference_sequential_13_layer_call_and_return_conditional_losses_970610

inputsB
'conv2d_9_conv2d_readvariableop_resource:?d<
.batch_normalization_13_readvariableop_resource:d>
0batch_normalization_13_readvariableop_1_resource:dM
?batch_normalization_13_fusedbatchnormv3_readvariableop_resource:dO
Abatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:d
identity??%batch_normalization_13/AssignNewValue?'batch_normalization_13/AssignNewValue_1?6batch_normalization_13/FusedBatchNormV3/ReadVariableOp?8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_13/ReadVariableOp?'batch_normalization_13/ReadVariableOp_1?conv2d_9/Conv2D/ReadVariableOp?
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*'
_output_shapes
:?d*
dtype0?
conv2d_9/Conv2DConv2Dinputs&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d*
paddingSAME*
strides
?
%batch_normalization_13/ReadVariableOpReadVariableOp.batch_normalization_13_readvariableop_resource*
_output_shapes
:d*
dtype0?
'batch_normalization_13/ReadVariableOp_1ReadVariableOp0batch_normalization_13_readvariableop_1_resource*
_output_shapes
:d*
dtype0?
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:d*
dtype0?
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:d*
dtype0?
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3conv2d_9/Conv2D:output:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????d:d:d:d:d:*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_13/AssignNewValueAssignVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource4batch_normalization_13/FusedBatchNormV3:batch_mean:07^batch_normalization_13/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization_13/AssignNewValue_1AssignVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_13/FusedBatchNormV3:batch_variance:09^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_re_lu_9/LeakyRelu	LeakyRelu+batch_normalization_13/FusedBatchNormV3:y:0*/
_output_shapes
:?????????d|
IdentityIdentity%leaky_re_lu_9/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????d?
NoOpNoOp&^batch_normalization_13/AssignNewValue(^batch_normalization_13/AssignNewValue_17^batch_normalization_13/FusedBatchNormV3/ReadVariableOp9^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_13/ReadVariableOp(^batch_normalization_13/ReadVariableOp_1^conv2d_9/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 2N
%batch_normalization_13/AssignNewValue%batch_normalization_13/AssignNewValue2R
'batch_normalization_13/AssignNewValue_1'batch_normalization_13/AssignNewValue_12p
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp6batch_normalization_13/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_18batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_13/ReadVariableOp%batch_normalization_13/ReadVariableOp2R
'batch_normalization_13/ReadVariableOp_1'batch_normalization_13/ReadVariableOp_12@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
H__inference_conv_block_9_layer_call_and_return_conditional_losses_968971
input_1/
sequential_13_968959:?d"
sequential_13_968961:d"
sequential_13_968963:d"
sequential_13_968965:d"
sequential_13_968967:d
identity??%sequential_13/StatefulPartitionedCall?
%sequential_13/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_13_968959sequential_13_968961sequential_13_968963sequential_13_968965sequential_13_968967*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_13_layer_call_and_return_conditional_losses_968787?
IdentityIdentity.sequential_13/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????dn
NoOpNoOp&^sequential_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 2N
%sequential_13/StatefulPartitionedCall%sequential_13/StatefulPartitionedCall:Y U
0
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
.__inference_sequential_10_layer_call_fn_970329

inputs"
unknown:@?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_967464x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@@@: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
I__inference_sequential_10_layer_call_and_return_conditional_losses_967614
conv2d_6_input*
conv2d_6_967600:@?,
batch_normalization_10_967603:	?,
batch_normalization_10_967605:	?,
batch_normalization_10_967607:	?,
batch_normalization_10_967609:	?
identity??.batch_normalization_10/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallconv2d_6_inputconv2d_6_967600*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_967425?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_10_967603batch_normalization_10_967605batch_normalization_10_967607batch_normalization_10_967609*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_967446?
leaky_re_lu_6/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_967461~
IdentityIdentity&leaky_re_lu_6/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ??
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@@@: : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????@@@
(
_user_specified_nameconv2d_6_input
?
?
I__inference_sequential_12_layer_call_and_return_conditional_losses_968443
conv2d_8_input+
conv2d_8_968429:??,
batch_normalization_12_968432:	?,
batch_normalization_12_968434:	?,
batch_normalization_12_968436:	?,
batch_normalization_12_968438:	?
identity??.batch_normalization_12/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputconv2d_8_968429*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_968237?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_12_968432batch_normalization_12_968434batch_normalization_12_968436batch_normalization_12_968438*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_968328?
leaky_re_lu_8/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_968273~
IdentityIdentity&leaky_re_lu_8/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp/^batch_normalization_12/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:` \
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_8_input
?

?
H__inference_conv_block_6_layer_call_and_return_conditional_losses_967753
input_1/
sequential_10_967741:@?#
sequential_10_967743:	?#
sequential_10_967745:	?#
sequential_10_967747:	?#
sequential_10_967749:	?
identity??%sequential_10/StatefulPartitionedCall?
%sequential_10/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_10_967741sequential_10_967743sequential_10_967745sequential_10_967747sequential_10_967749*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_967569?
IdentityIdentity.sequential_10/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?n
NoOpNoOp&^sequential_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@@@: : : : : 2N
%sequential_10/StatefulPartitionedCall%sequential_10/StatefulPartitionedCall:X T
/
_output_shapes
:?????????@@@
!
_user_specified_name	input_1
??
?(
!__inference__wrapped_model_966941
input_1i
Odiscriminator_conv_block_5_sequential_9_conv2d_5_conv2d_readvariableop_resource:@c
Udiscriminator_conv_block_5_sequential_9_batch_normalization_9_readvariableop_resource:@e
Wdiscriminator_conv_block_5_sequential_9_batch_normalization_9_readvariableop_1_resource:@t
fdiscriminator_conv_block_5_sequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:@v
hdiscriminator_conv_block_5_sequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:@k
Pdiscriminator_conv_block_6_sequential_10_conv2d_6_conv2d_readvariableop_resource:@?f
Wdiscriminator_conv_block_6_sequential_10_batch_normalization_10_readvariableop_resource:	?h
Ydiscriminator_conv_block_6_sequential_10_batch_normalization_10_readvariableop_1_resource:	?w
hdiscriminator_conv_block_6_sequential_10_batch_normalization_10_fusedbatchnormv3_readvariableop_resource:	?y
jdiscriminator_conv_block_6_sequential_10_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:	?l
Pdiscriminator_conv_block_7_sequential_11_conv2d_7_conv2d_readvariableop_resource:??f
Wdiscriminator_conv_block_7_sequential_11_batch_normalization_11_readvariableop_resource:	?h
Ydiscriminator_conv_block_7_sequential_11_batch_normalization_11_readvariableop_1_resource:	?w
hdiscriminator_conv_block_7_sequential_11_batch_normalization_11_fusedbatchnormv3_readvariableop_resource:	?y
jdiscriminator_conv_block_7_sequential_11_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:	?l
Pdiscriminator_conv_block_8_sequential_12_conv2d_8_conv2d_readvariableop_resource:??f
Wdiscriminator_conv_block_8_sequential_12_batch_normalization_12_readvariableop_resource:	?h
Ydiscriminator_conv_block_8_sequential_12_batch_normalization_12_readvariableop_1_resource:	?w
hdiscriminator_conv_block_8_sequential_12_batch_normalization_12_fusedbatchnormv3_readvariableop_resource:	?y
jdiscriminator_conv_block_8_sequential_12_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:	?k
Pdiscriminator_conv_block_9_sequential_13_conv2d_9_conv2d_readvariableop_resource:?de
Wdiscriminator_conv_block_9_sequential_13_batch_normalization_13_readvariableop_resource:dg
Ydiscriminator_conv_block_9_sequential_13_batch_normalization_13_readvariableop_1_resource:dv
hdiscriminator_conv_block_9_sequential_13_batch_normalization_13_fusedbatchnormv3_readvariableop_resource:dx
jdiscriminator_conv_block_9_sequential_13_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:dP
6discriminator_conv2d_10_conv2d_readvariableop_resource:d
identity

identity_1??-discriminator/conv2d_10/Conv2D/ReadVariableOp?]discriminator/conv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp?_discriminator/conv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?Ldiscriminator/conv_block_5/sequential_9/batch_normalization_9/ReadVariableOp?Ndiscriminator/conv_block_5/sequential_9/batch_normalization_9/ReadVariableOp_1?Fdiscriminator/conv_block_5/sequential_9/conv2d_5/Conv2D/ReadVariableOp?_discriminator/conv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp?adiscriminator/conv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?Ndiscriminator/conv_block_6/sequential_10/batch_normalization_10/ReadVariableOp?Pdiscriminator/conv_block_6/sequential_10/batch_normalization_10/ReadVariableOp_1?Gdiscriminator/conv_block_6/sequential_10/conv2d_6/Conv2D/ReadVariableOp?_discriminator/conv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp?adiscriminator/conv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1?Ndiscriminator/conv_block_7/sequential_11/batch_normalization_11/ReadVariableOp?Pdiscriminator/conv_block_7/sequential_11/batch_normalization_11/ReadVariableOp_1?Gdiscriminator/conv_block_7/sequential_11/conv2d_7/Conv2D/ReadVariableOp?_discriminator/conv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp?adiscriminator/conv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?Ndiscriminator/conv_block_8/sequential_12/batch_normalization_12/ReadVariableOp?Pdiscriminator/conv_block_8/sequential_12/batch_normalization_12/ReadVariableOp_1?Gdiscriminator/conv_block_8/sequential_12/conv2d_8/Conv2D/ReadVariableOp?_discriminator/conv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp?adiscriminator/conv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?Ndiscriminator/conv_block_9/sequential_13/batch_normalization_13/ReadVariableOp?Pdiscriminator/conv_block_9/sequential_13/batch_normalization_13/ReadVariableOp_1?Gdiscriminator/conv_block_9/sequential_13/conv2d_9/Conv2D/ReadVariableOp?
Fdiscriminator/conv_block_5/sequential_9/conv2d_5/Conv2D/ReadVariableOpReadVariableOpOdiscriminator_conv_block_5_sequential_9_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
7discriminator/conv_block_5/sequential_9/conv2d_5/Conv2DConv2Dinput_1Ndiscriminator/conv_block_5/sequential_9/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
Ldiscriminator/conv_block_5/sequential_9/batch_normalization_9/ReadVariableOpReadVariableOpUdiscriminator_conv_block_5_sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:@*
dtype0?
Ndiscriminator/conv_block_5/sequential_9/batch_normalization_9/ReadVariableOp_1ReadVariableOpWdiscriminator_conv_block_5_sequential_9_batch_normalization_9_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
]discriminator/conv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpfdiscriminator_conv_block_5_sequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
_discriminator/conv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOphdiscriminator_conv_block_5_sequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
Ndiscriminator/conv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3@discriminator/conv_block_5/sequential_9/conv2d_5/Conv2D:output:0Tdiscriminator/conv_block_5/sequential_9/batch_normalization_9/ReadVariableOp:value:0Vdiscriminator/conv_block_5/sequential_9/batch_normalization_9/ReadVariableOp_1:value:0ediscriminator/conv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0gdiscriminator/conv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
is_training( ?
?discriminator/conv_block_5/sequential_9/leaky_re_lu_5/LeakyRelu	LeakyReluRdiscriminator/conv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@@@?
Gdiscriminator/conv_block_6/sequential_10/conv2d_6/Conv2D/ReadVariableOpReadVariableOpPdiscriminator_conv_block_6_sequential_10_conv2d_6_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
8discriminator/conv_block_6/sequential_10/conv2d_6/Conv2DConv2DMdiscriminator/conv_block_5/sequential_9/leaky_re_lu_5/LeakyRelu:activations:0Odiscriminator/conv_block_6/sequential_10/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
Ndiscriminator/conv_block_6/sequential_10/batch_normalization_10/ReadVariableOpReadVariableOpWdiscriminator_conv_block_6_sequential_10_batch_normalization_10_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Pdiscriminator/conv_block_6/sequential_10/batch_normalization_10/ReadVariableOp_1ReadVariableOpYdiscriminator_conv_block_6_sequential_10_batch_normalization_10_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
_discriminator/conv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOphdiscriminator_conv_block_6_sequential_10_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
adiscriminator/conv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjdiscriminator_conv_block_6_sequential_10_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Pdiscriminator/conv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3Adiscriminator/conv_block_6/sequential_10/conv2d_6/Conv2D:output:0Vdiscriminator/conv_block_6/sequential_10/batch_normalization_10/ReadVariableOp:value:0Xdiscriminator/conv_block_6/sequential_10/batch_normalization_10/ReadVariableOp_1:value:0gdiscriminator/conv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0idiscriminator/conv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
is_training( ?
@discriminator/conv_block_6/sequential_10/leaky_re_lu_6/LeakyRelu	LeakyReluTdiscriminator/conv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3:y:0*0
_output_shapes
:?????????  ??
Gdiscriminator/conv_block_7/sequential_11/conv2d_7/Conv2D/ReadVariableOpReadVariableOpPdiscriminator_conv_block_7_sequential_11_conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
8discriminator/conv_block_7/sequential_11/conv2d_7/Conv2DConv2DNdiscriminator/conv_block_6/sequential_10/leaky_re_lu_6/LeakyRelu:activations:0Odiscriminator/conv_block_7/sequential_11/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
Ndiscriminator/conv_block_7/sequential_11/batch_normalization_11/ReadVariableOpReadVariableOpWdiscriminator_conv_block_7_sequential_11_batch_normalization_11_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Pdiscriminator/conv_block_7/sequential_11/batch_normalization_11/ReadVariableOp_1ReadVariableOpYdiscriminator_conv_block_7_sequential_11_batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
_discriminator/conv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOphdiscriminator_conv_block_7_sequential_11_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
adiscriminator/conv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjdiscriminator_conv_block_7_sequential_11_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Pdiscriminator/conv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3Adiscriminator/conv_block_7/sequential_11/conv2d_7/Conv2D:output:0Vdiscriminator/conv_block_7/sequential_11/batch_normalization_11/ReadVariableOp:value:0Xdiscriminator/conv_block_7/sequential_11/batch_normalization_11/ReadVariableOp_1:value:0gdiscriminator/conv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0idiscriminator/conv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( ?
@discriminator/conv_block_7/sequential_11/leaky_re_lu_7/LeakyRelu	LeakyReluTdiscriminator/conv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3:y:0*0
_output_shapes
:???????????
Gdiscriminator/conv_block_8/sequential_12/conv2d_8/Conv2D/ReadVariableOpReadVariableOpPdiscriminator_conv_block_8_sequential_12_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
8discriminator/conv_block_8/sequential_12/conv2d_8/Conv2DConv2DNdiscriminator/conv_block_7/sequential_11/leaky_re_lu_7/LeakyRelu:activations:0Odiscriminator/conv_block_8/sequential_12/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
Ndiscriminator/conv_block_8/sequential_12/batch_normalization_12/ReadVariableOpReadVariableOpWdiscriminator_conv_block_8_sequential_12_batch_normalization_12_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Pdiscriminator/conv_block_8/sequential_12/batch_normalization_12/ReadVariableOp_1ReadVariableOpYdiscriminator_conv_block_8_sequential_12_batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
_discriminator/conv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOphdiscriminator_conv_block_8_sequential_12_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
adiscriminator/conv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjdiscriminator_conv_block_8_sequential_12_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Pdiscriminator/conv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3Adiscriminator/conv_block_8/sequential_12/conv2d_8/Conv2D:output:0Vdiscriminator/conv_block_8/sequential_12/batch_normalization_12/ReadVariableOp:value:0Xdiscriminator/conv_block_8/sequential_12/batch_normalization_12/ReadVariableOp_1:value:0gdiscriminator/conv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0idiscriminator/conv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( ?
@discriminator/conv_block_8/sequential_12/leaky_re_lu_8/LeakyRelu	LeakyReluTdiscriminator/conv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3:y:0*0
_output_shapes
:???????????
Gdiscriminator/conv_block_9/sequential_13/conv2d_9/Conv2D/ReadVariableOpReadVariableOpPdiscriminator_conv_block_9_sequential_13_conv2d_9_conv2d_readvariableop_resource*'
_output_shapes
:?d*
dtype0?
8discriminator/conv_block_9/sequential_13/conv2d_9/Conv2DConv2DNdiscriminator/conv_block_8/sequential_12/leaky_re_lu_8/LeakyRelu:activations:0Odiscriminator/conv_block_9/sequential_13/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d*
paddingSAME*
strides
?
Ndiscriminator/conv_block_9/sequential_13/batch_normalization_13/ReadVariableOpReadVariableOpWdiscriminator_conv_block_9_sequential_13_batch_normalization_13_readvariableop_resource*
_output_shapes
:d*
dtype0?
Pdiscriminator/conv_block_9/sequential_13/batch_normalization_13/ReadVariableOp_1ReadVariableOpYdiscriminator_conv_block_9_sequential_13_batch_normalization_13_readvariableop_1_resource*
_output_shapes
:d*
dtype0?
_discriminator/conv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOphdiscriminator_conv_block_9_sequential_13_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:d*
dtype0?
adiscriminator/conv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjdiscriminator_conv_block_9_sequential_13_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:d*
dtype0?
Pdiscriminator/conv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3FusedBatchNormV3Adiscriminator/conv_block_9/sequential_13/conv2d_9/Conv2D:output:0Vdiscriminator/conv_block_9/sequential_13/batch_normalization_13/ReadVariableOp:value:0Xdiscriminator/conv_block_9/sequential_13/batch_normalization_13/ReadVariableOp_1:value:0gdiscriminator/conv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0idiscriminator/conv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????d:d:d:d:d:*
epsilon%o?:*
is_training( ?
@discriminator/conv_block_9/sequential_13/leaky_re_lu_9/LeakyRelu	LeakyReluTdiscriminator/conv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3:y:0*/
_output_shapes
:?????????d?
-discriminator/conv2d_10/Conv2D/ReadVariableOpReadVariableOp6discriminator_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:d*
dtype0?
discriminator/conv2d_10/Conv2DConv2DNdiscriminator/conv_block_9/sequential_13/leaky_re_lu_9/LeakyRelu:activations:05discriminator/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
discriminator/conv2d_10/SigmoidSigmoid'discriminator/conv2d_10/Conv2D:output:0*
T0*/
_output_shapes
:?????????z
IdentityIdentity#discriminator/conv2d_10/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:??????????

Identity_1IdentityNdiscriminator/conv_block_9/sequential_13/leaky_re_lu_9/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????d?
NoOpNoOp.^discriminator/conv2d_10/Conv2D/ReadVariableOp^^discriminator/conv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp`^discriminator/conv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1M^discriminator/conv_block_5/sequential_9/batch_normalization_9/ReadVariableOpO^discriminator/conv_block_5/sequential_9/batch_normalization_9/ReadVariableOp_1G^discriminator/conv_block_5/sequential_9/conv2d_5/Conv2D/ReadVariableOp`^discriminator/conv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOpb^discriminator/conv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1O^discriminator/conv_block_6/sequential_10/batch_normalization_10/ReadVariableOpQ^discriminator/conv_block_6/sequential_10/batch_normalization_10/ReadVariableOp_1H^discriminator/conv_block_6/sequential_10/conv2d_6/Conv2D/ReadVariableOp`^discriminator/conv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOpb^discriminator/conv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1O^discriminator/conv_block_7/sequential_11/batch_normalization_11/ReadVariableOpQ^discriminator/conv_block_7/sequential_11/batch_normalization_11/ReadVariableOp_1H^discriminator/conv_block_7/sequential_11/conv2d_7/Conv2D/ReadVariableOp`^discriminator/conv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOpb^discriminator/conv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1O^discriminator/conv_block_8/sequential_12/batch_normalization_12/ReadVariableOpQ^discriminator/conv_block_8/sequential_12/batch_normalization_12/ReadVariableOp_1H^discriminator/conv_block_8/sequential_12/conv2d_8/Conv2D/ReadVariableOp`^discriminator/conv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOpb^discriminator/conv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1O^discriminator/conv_block_9/sequential_13/batch_normalization_13/ReadVariableOpQ^discriminator/conv_block_9/sequential_13/batch_normalization_13/ReadVariableOp_1H^discriminator/conv_block_9/sequential_13/conv2d_9/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-discriminator/conv2d_10/Conv2D/ReadVariableOp-discriminator/conv2d_10/Conv2D/ReadVariableOp2?
]discriminator/conv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp]discriminator/conv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2?
_discriminator/conv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1_discriminator/conv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12?
Ldiscriminator/conv_block_5/sequential_9/batch_normalization_9/ReadVariableOpLdiscriminator/conv_block_5/sequential_9/batch_normalization_9/ReadVariableOp2?
Ndiscriminator/conv_block_5/sequential_9/batch_normalization_9/ReadVariableOp_1Ndiscriminator/conv_block_5/sequential_9/batch_normalization_9/ReadVariableOp_12?
Fdiscriminator/conv_block_5/sequential_9/conv2d_5/Conv2D/ReadVariableOpFdiscriminator/conv_block_5/sequential_9/conv2d_5/Conv2D/ReadVariableOp2?
_discriminator/conv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_discriminator/conv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp2?
adiscriminator/conv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1adiscriminator/conv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12?
Ndiscriminator/conv_block_6/sequential_10/batch_normalization_10/ReadVariableOpNdiscriminator/conv_block_6/sequential_10/batch_normalization_10/ReadVariableOp2?
Pdiscriminator/conv_block_6/sequential_10/batch_normalization_10/ReadVariableOp_1Pdiscriminator/conv_block_6/sequential_10/batch_normalization_10/ReadVariableOp_12?
Gdiscriminator/conv_block_6/sequential_10/conv2d_6/Conv2D/ReadVariableOpGdiscriminator/conv_block_6/sequential_10/conv2d_6/Conv2D/ReadVariableOp2?
_discriminator/conv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_discriminator/conv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp2?
adiscriminator/conv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1adiscriminator/conv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12?
Ndiscriminator/conv_block_7/sequential_11/batch_normalization_11/ReadVariableOpNdiscriminator/conv_block_7/sequential_11/batch_normalization_11/ReadVariableOp2?
Pdiscriminator/conv_block_7/sequential_11/batch_normalization_11/ReadVariableOp_1Pdiscriminator/conv_block_7/sequential_11/batch_normalization_11/ReadVariableOp_12?
Gdiscriminator/conv_block_7/sequential_11/conv2d_7/Conv2D/ReadVariableOpGdiscriminator/conv_block_7/sequential_11/conv2d_7/Conv2D/ReadVariableOp2?
_discriminator/conv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_discriminator/conv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp2?
adiscriminator/conv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1adiscriminator/conv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12?
Ndiscriminator/conv_block_8/sequential_12/batch_normalization_12/ReadVariableOpNdiscriminator/conv_block_8/sequential_12/batch_normalization_12/ReadVariableOp2?
Pdiscriminator/conv_block_8/sequential_12/batch_normalization_12/ReadVariableOp_1Pdiscriminator/conv_block_8/sequential_12/batch_normalization_12/ReadVariableOp_12?
Gdiscriminator/conv_block_8/sequential_12/conv2d_8/Conv2D/ReadVariableOpGdiscriminator/conv_block_8/sequential_12/conv2d_8/Conv2D/ReadVariableOp2?
_discriminator/conv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_discriminator/conv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp2?
adiscriminator/conv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1adiscriminator/conv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12?
Ndiscriminator/conv_block_9/sequential_13/batch_normalization_13/ReadVariableOpNdiscriminator/conv_block_9/sequential_13/batch_normalization_13/ReadVariableOp2?
Pdiscriminator/conv_block_9/sequential_13/batch_normalization_13/ReadVariableOp_1Pdiscriminator/conv_block_9/sequential_13/batch_normalization_13/ReadVariableOp_12?
Gdiscriminator/conv_block_9/sequential_13/conv2d_9/Conv2D/ReadVariableOpGdiscriminator/conv_block_9/sequential_13/conv2d_9/Conv2D/ReadVariableOp:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
-__inference_conv_block_9_layer_call_fn_970181

inputs"
unknown:?d
	unknown_0:d
	unknown_1:d
	unknown_2:d
	unknown_3:d
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_9_layer_call_and_return_conditional_losses_968913w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_12_layer_call_and_return_conditional_losses_968381

inputs+
conv2d_8_968367:??,
batch_normalization_12_968370:	?,
batch_normalization_12_968372:	?,
batch_normalization_12_968374:	?,
batch_normalization_12_968376:	?
identity??.batch_normalization_12/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8_968367*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_968237?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_12_968370batch_normalization_12_968372batch_normalization_12_968374batch_normalization_12_968376*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_968328?
leaky_re_lu_8/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_968273~
IdentityIdentity&leaky_re_lu_8/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp/^batch_normalization_12/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?!
?
I__inference_sequential_10_layer_call_and_return_conditional_losses_970388

inputsB
'conv2d_6_conv2d_readvariableop_resource:@?=
.batch_normalization_10_readvariableop_resource:	??
0batch_normalization_10_readvariableop_1_resource:	?N
?batch_normalization_10_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:	?
identity??%batch_normalization_10/AssignNewValue?'batch_normalization_10/AssignNewValue_1?6batch_normalization_10/FusedBatchNormV3/ReadVariableOp?8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_10/ReadVariableOp?'batch_normalization_10/ReadVariableOp_1?conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_6/Conv2D:output:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_10/AssignNewValueAssignVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource4batch_normalization_10/FusedBatchNormV3:batch_mean:07^batch_normalization_10/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization_10/AssignNewValue_1AssignVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_10/FusedBatchNormV3:batch_variance:09^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_re_lu_6/LeakyRelu	LeakyRelu+batch_normalization_10/FusedBatchNormV3:y:0*0
_output_shapes
:?????????  ?}
IdentityIdentity%leaky_re_lu_6/LeakyRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????  ??
NoOpNoOp&^batch_normalization_10/AssignNewValue(^batch_normalization_10/AssignNewValue_17^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_1^conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@@@: : : : : 2N
%batch_normalization_10/AssignNewValue%batch_normalization_10/AssignNewValue2R
'batch_normalization_10/AssignNewValue_1'batch_normalization_10/AssignNewValue_12p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_971322

inputs%
readvariableop_resource:d'
readvariableop_1_resource:d6
(fusedbatchnormv3_readvariableop_resource:d8
*fusedbatchnormv3_readvariableop_1_resource:d
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:d*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:d*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:d*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:d*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????d:d:d:d:d:*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????d?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????d: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????d
 
_user_specified_nameinputs
?	
?
-__inference_sequential_9_layer_call_fn_967191
conv2d_5_input!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_967163w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
1
_output_shapes
:???????????
(
_user_specified_nameconv2d_5_input
?
?
D__inference_conv2d_9_layer_call_and_return_conditional_losses_968643

inputs9
conv2d_readvariableop_resource:?d
identity??Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?d*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d*
paddingSAME*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:?????????d^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_967225
conv2d_5_input)
conv2d_5_967211:@*
batch_normalization_9_967214:@*
batch_normalization_9_967216:@*
batch_normalization_9_967218:@*
batch_normalization_9_967220:@
identity??-batch_normalization_9/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputconv2d_5_967211*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_967019?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_9_967214batch_normalization_9_967216batch_normalization_9_967218batch_normalization_9_967220*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_967110?
leaky_re_lu_5/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_967055}
IdentityIdentity&leaky_re_lu_5/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@?
NoOpNoOp.^batch_normalization_9/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????: : : : : 2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall:a ]
1
_output_shapes
:???????????
(
_user_specified_nameconv2d_5_input
?;
?
__inference__traced_save_971452
file_prefix=
9savev2_discriminator_conv2d_10_kernel_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop:
6savev2_batch_normalization_9_gamma_read_readvariableop9
5savev2_batch_normalization_9_beta_read_readvariableop@
<savev2_batch_normalization_9_moving_mean_read_readvariableopD
@savev2_batch_normalization_9_moving_variance_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop;
7savev2_batch_normalization_10_gamma_read_readvariableop:
6savev2_batch_normalization_10_beta_read_readvariableopA
=savev2_batch_normalization_10_moving_mean_read_readvariableopE
Asavev2_batch_normalization_10_moving_variance_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop;
7savev2_batch_normalization_11_gamma_read_readvariableop:
6savev2_batch_normalization_11_beta_read_readvariableopA
=savev2_batch_normalization_11_moving_mean_read_readvariableopE
Asavev2_batch_normalization_11_moving_variance_read_readvariableop.
*savev2_conv2d_8_kernel_read_readvariableop;
7savev2_batch_normalization_12_gamma_read_readvariableop:
6savev2_batch_normalization_12_beta_read_readvariableopA
=savev2_batch_normalization_12_moving_mean_read_readvariableopE
Asavev2_batch_normalization_12_moving_variance_read_readvariableop.
*savev2_conv2d_9_kernel_read_readvariableop;
7savev2_batch_normalization_13_gamma_read_readvariableop:
6savev2_batch_normalization_13_beta_read_readvariableopA
=savev2_batch_normalization_13_moving_mean_read_readvariableopE
Asavev2_batch_normalization_13_moving_variance_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
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
_temp/part?
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
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B)outputs/kernel/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:09savev2_discriminator_conv2d_10_kernel_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop6savev2_batch_normalization_9_gamma_read_readvariableop5savev2_batch_normalization_9_beta_read_readvariableop<savev2_batch_normalization_9_moving_mean_read_readvariableop@savev2_batch_normalization_9_moving_variance_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop7savev2_batch_normalization_10_gamma_read_readvariableop6savev2_batch_normalization_10_beta_read_readvariableop=savev2_batch_normalization_10_moving_mean_read_readvariableopAsavev2_batch_normalization_10_moving_variance_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop7savev2_batch_normalization_11_gamma_read_readvariableop6savev2_batch_normalization_11_beta_read_readvariableop=savev2_batch_normalization_11_moving_mean_read_readvariableopAsavev2_batch_normalization_11_moving_variance_read_readvariableop*savev2_conv2d_8_kernel_read_readvariableop7savev2_batch_normalization_12_gamma_read_readvariableop6savev2_batch_normalization_12_beta_read_readvariableop=savev2_batch_normalization_12_moving_mean_read_readvariableopAsavev2_batch_normalization_12_moving_variance_read_readvariableop*savev2_conv2d_9_kernel_read_readvariableop7savev2_batch_normalization_13_gamma_read_readvariableop6savev2_batch_normalization_13_beta_read_readvariableop=savev2_batch_normalization_13_moving_mean_read_readvariableopAsavev2_batch_normalization_13_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *)
dtypes
2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :d:@:@:@:@:@:@?:?:?:?:?:??:?:?:?:?:??:?:?:?:?:?d:d:d:d:d: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:d:,(
&
_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:!	

_output_shapes	
:?:!


_output_shapes	
:?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:-)
'
_output_shapes
:?d: 

_output_shapes
:d: 

_output_shapes
:d: 

_output_shapes
:d: 

_output_shapes
:d:

_output_shapes
: 
?
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_967163

inputs)
conv2d_5_967149:@*
batch_normalization_9_967152:@*
batch_normalization_9_967154:@*
batch_normalization_9_967156:@*
batch_normalization_9_967158:@
identity??-batch_normalization_9/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_967149*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_967019?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_9_967152batch_normalization_9_967154batch_normalization_9_967156batch_normalization_9_967158*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_967110?
leaky_re_lu_5/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_967055}
IdentityIdentity&leaky_re_lu_5/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@?
NoOpNoOp.^batch_normalization_9/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????: : : : : 2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
-__inference_conv_block_7_layer_call_fn_970018

inputs#
unknown:??
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_7_layer_call_and_return_conditional_losses_968056x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????  ?: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?

?
H__inference_conv_block_9_layer_call_and_return_conditional_losses_968868

inputs/
sequential_13_968856:?d"
sequential_13_968858:d"
sequential_13_968860:d"
sequential_13_968862:d"
sequential_13_968864:d
identity??%sequential_13/StatefulPartitionedCall?
%sequential_13/StatefulPartitionedCallStatefulPartitionedCallinputssequential_13_968856sequential_13_968858sequential_13_968860sequential_13_968862sequential_13_968864*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_13_layer_call_and_return_conditional_losses_968682?
IdentityIdentity.sequential_13/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????dn
NoOpNoOp&^sequential_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 2N
%sequential_13/StatefulPartitionedCall%sequential_13/StatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?(
?
H__inference_conv_block_9_layer_call_and_return_conditional_losses_970225

inputsP
5sequential_13_conv2d_9_conv2d_readvariableop_resource:?dJ
<sequential_13_batch_normalization_13_readvariableop_resource:dL
>sequential_13_batch_normalization_13_readvariableop_1_resource:d[
Msequential_13_batch_normalization_13_fusedbatchnormv3_readvariableop_resource:d]
Osequential_13_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:d
identity??3sequential_13/batch_normalization_13/AssignNewValue?5sequential_13/batch_normalization_13/AssignNewValue_1?Dsequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp?Fsequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?3sequential_13/batch_normalization_13/ReadVariableOp?5sequential_13/batch_normalization_13/ReadVariableOp_1?,sequential_13/conv2d_9/Conv2D/ReadVariableOp?
,sequential_13/conv2d_9/Conv2D/ReadVariableOpReadVariableOp5sequential_13_conv2d_9_conv2d_readvariableop_resource*'
_output_shapes
:?d*
dtype0?
sequential_13/conv2d_9/Conv2DConv2Dinputs4sequential_13/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d*
paddingSAME*
strides
?
3sequential_13/batch_normalization_13/ReadVariableOpReadVariableOp<sequential_13_batch_normalization_13_readvariableop_resource*
_output_shapes
:d*
dtype0?
5sequential_13/batch_normalization_13/ReadVariableOp_1ReadVariableOp>sequential_13_batch_normalization_13_readvariableop_1_resource*
_output_shapes
:d*
dtype0?
Dsequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_13_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:d*
dtype0?
Fsequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_13_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:d*
dtype0?
5sequential_13/batch_normalization_13/FusedBatchNormV3FusedBatchNormV3&sequential_13/conv2d_9/Conv2D:output:0;sequential_13/batch_normalization_13/ReadVariableOp:value:0=sequential_13/batch_normalization_13/ReadVariableOp_1:value:0Lsequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????d:d:d:d:d:*
epsilon%o?:*
exponential_avg_factor%
?#<?
3sequential_13/batch_normalization_13/AssignNewValueAssignVariableOpMsequential_13_batch_normalization_13_fusedbatchnormv3_readvariableop_resourceBsequential_13/batch_normalization_13/FusedBatchNormV3:batch_mean:0E^sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
5sequential_13/batch_normalization_13/AssignNewValue_1AssignVariableOpOsequential_13_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resourceFsequential_13/batch_normalization_13/FusedBatchNormV3:batch_variance:0G^sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
%sequential_13/leaky_re_lu_9/LeakyRelu	LeakyRelu9sequential_13/batch_normalization_13/FusedBatchNormV3:y:0*/
_output_shapes
:?????????d?
IdentityIdentity3sequential_13/leaky_re_lu_9/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????d?
NoOpNoOp4^sequential_13/batch_normalization_13/AssignNewValue6^sequential_13/batch_normalization_13/AssignNewValue_1E^sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOpG^sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_14^sequential_13/batch_normalization_13/ReadVariableOp6^sequential_13/batch_normalization_13/ReadVariableOp_1-^sequential_13/conv2d_9/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 2j
3sequential_13/batch_normalization_13/AssignNewValue3sequential_13/batch_normalization_13/AssignNewValue2n
5sequential_13/batch_normalization_13/AssignNewValue_15sequential_13/batch_normalization_13/AssignNewValue_12?
Dsequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOpDsequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp2?
Fsequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1Fsequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12j
3sequential_13/batch_normalization_13/ReadVariableOp3sequential_13/batch_normalization_13/ReadVariableOp2n
5sequential_13/batch_normalization_13/ReadVariableOp_15sequential_13/batch_normalization_13/ReadVariableOp_12\
,sequential_13/conv2d_9/Conv2D/ReadVariableOp,sequential_13/conv2d_9/Conv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_967922

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
H__inference_conv_block_9_layer_call_and_return_conditional_losses_968913

inputs/
sequential_13_968901:?d"
sequential_13_968903:d"
sequential_13_968905:d"
sequential_13_968907:d"
sequential_13_968909:d
identity??%sequential_13/StatefulPartitionedCall?
%sequential_13/StatefulPartitionedCallStatefulPartitionedCallinputssequential_13_968901sequential_13_968903sequential_13_968905sequential_13_968907sequential_13_968909*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_13_layer_call_and_return_conditional_losses_968787?
IdentityIdentity.sequential_13/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????dn
NoOpNoOp&^sequential_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 2N
%sequential_13/StatefulPartitionedCall%sequential_13/StatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
.__inference_sequential_13_layer_call_fn_968815
conv2d_9_input"
unknown:?d
	unknown_0:d
	unknown_1:d
	unknown_2:d
	unknown_3:d
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_9_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_13_layer_call_and_return_conditional_losses_968787w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_9_input
?
?
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_971286

inputs%
readvariableop_resource:d'
readvariableop_1_resource:d6
(fusedbatchnormv3_readvariableop_resource:d8
*fusedbatchnormv3_readvariableop_1_resource:d
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:d*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:d*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:d*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:d*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????d:d:d:d:d:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????d?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????d: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????d
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_968679

inputs
identityO
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????dg
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????d:W S
/
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
E__inference_conv2d_10_layer_call_and_return_conditional_losses_970240

inputs8
conv2d_readvariableop_resource:d
identity??Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:d*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
]
SigmoidSigmoidConv2D:output:0*
T0*/
_output_shapes
:?????????b
IdentityIdentitySigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????d: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????d
 
_user_specified_nameinputs
?	
?
-__inference_sequential_9_layer_call_fn_967071
conv2d_5_input!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_967058w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
1
_output_shapes
:???????????
(
_user_specified_nameconv2d_5_input
?
?
-__inference_conv_block_9_layer_call_fn_970166

inputs"
unknown:?d
	unknown_0:d
	unknown_1:d
	unknown_2:d
	unknown_3:d
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_9_layer_call_and_return_conditional_losses_968868w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_8_layer_call_and_return_conditional_losses_971068

inputs:
conv2d_readvariableop_resource:??
identity??Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:??????????^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
H__inference_conv_block_7_layer_call_and_return_conditional_losses_968101

inputs0
sequential_11_968089:??#
sequential_11_968091:	?#
sequential_11_968093:	?#
sequential_11_968095:	?#
sequential_11_968097:	?
identity??%sequential_11/StatefulPartitionedCall?
%sequential_11/StatefulPartitionedCallStatefulPartitionedCallinputssequential_11_968089sequential_11_968091sequential_11_968093sequential_11_968095sequential_11_968097*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_11_layer_call_and_return_conditional_losses_967975?
IdentityIdentity.sequential_11/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????n
NoOpNoOp&^sequential_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????  ?: : : : : 2N
%sequential_11/StatefulPartitionedCall%sequential_11/StatefulPartitionedCall:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
E__inference_conv2d_10_layer_call_and_return_conditional_losses_969041

inputs8
conv2d_readvariableop_resource:d
identity??Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:d*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
]
SigmoidSigmoidConv2D:output:0*
T0*/
_output_shapes
:?????????b
IdentityIdentitySigmoid:y:0^NoOp*
T0*/
_output_shapes
:?????????^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????d: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_10_layer_call_fn_970824

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_967516x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????  ?: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_967446

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
is_training( l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????  ??
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????  ?: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
H__inference_conv_block_8_layer_call_and_return_conditional_losses_970129

inputsQ
5sequential_12_conv2d_8_conv2d_readvariableop_resource:??K
<sequential_12_batch_normalization_12_readvariableop_resource:	?M
>sequential_12_batch_normalization_12_readvariableop_1_resource:	?\
Msequential_12_batch_normalization_12_fusedbatchnormv3_readvariableop_resource:	?^
Osequential_12_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:	?
identity??Dsequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp?Fsequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?3sequential_12/batch_normalization_12/ReadVariableOp?5sequential_12/batch_normalization_12/ReadVariableOp_1?,sequential_12/conv2d_8/Conv2D/ReadVariableOp?
,sequential_12/conv2d_8/Conv2D/ReadVariableOpReadVariableOp5sequential_12_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
sequential_12/conv2d_8/Conv2DConv2Dinputs4sequential_12/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
3sequential_12/batch_normalization_12/ReadVariableOpReadVariableOp<sequential_12_batch_normalization_12_readvariableop_resource*
_output_shapes	
:?*
dtype0?
5sequential_12/batch_normalization_12/ReadVariableOp_1ReadVariableOp>sequential_12_batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Dsequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_12_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Fsequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_12_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
5sequential_12/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3&sequential_12/conv2d_8/Conv2D:output:0;sequential_12/batch_normalization_12/ReadVariableOp:value:0=sequential_12/batch_normalization_12/ReadVariableOp_1:value:0Lsequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( ?
%sequential_12/leaky_re_lu_8/LeakyRelu	LeakyRelu9sequential_12/batch_normalization_12/FusedBatchNormV3:y:0*0
_output_shapes
:???????????
IdentityIdentity3sequential_12/leaky_re_lu_8/LeakyRelu:activations:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOpE^sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOpG^sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_14^sequential_12/batch_normalization_12/ReadVariableOp6^sequential_12/batch_normalization_12/ReadVariableOp_1-^sequential_12/conv2d_8/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 2?
Dsequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOpDsequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp2?
Fsequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1Fsequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12j
3sequential_12/batch_normalization_12/ReadVariableOp3sequential_12/batch_normalization_12/ReadVariableOp2n
5sequential_12/batch_normalization_12/ReadVariableOp_15sequential_12/batch_normalization_12/ReadVariableOp_12\
,sequential_12/conv2d_8/Conv2D/ReadVariableOp,sequential_12/conv2d_8/Conv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_967369

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_9_layer_call_fn_970663

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_967040w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
.__inference_discriminator_layer_call_fn_969598

inputs!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@$
	unknown_4:@?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?&

unknown_14:??

unknown_15:	?

unknown_16:	?

unknown_17:	?

unknown_18:	?%

unknown_19:?d

unknown_20:d

unknown_21:d

unknown_22:d

unknown_23:d$

unknown_24:d
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:?????????:?????????d*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_discriminator_layer_call_and_return_conditional_losses_969047w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????y

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_970694

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_971350

inputs
identityO
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????dg
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????d:W S
/
_output_shapes
:?????????d
 
_user_specified_nameinputs
?

?
H__inference_conv_block_7_layer_call_and_return_conditional_losses_968056

inputs0
sequential_11_968044:??#
sequential_11_968046:	?#
sequential_11_968048:	?#
sequential_11_968050:	?#
sequential_11_968052:	?
identity??%sequential_11/StatefulPartitionedCall?
%sequential_11/StatefulPartitionedCallStatefulPartitionedCallinputssequential_11_968044sequential_11_968046sequential_11_968048sequential_11_968050sequential_11_968052*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_11_layer_call_and_return_conditional_losses_967870?
IdentityIdentity.sequential_11/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????n
NoOpNoOp&^sequential_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????  ?: : : : : 2N
%sequential_11/StatefulPartitionedCall%sequential_11/StatefulPartitionedCall:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
I__inference_sequential_10_layer_call_and_return_conditional_losses_970366

inputsB
'conv2d_6_conv2d_readvariableop_resource:@?=
.batch_normalization_10_readvariableop_resource:	??
0batch_normalization_10_readvariableop_1_resource:	?N
?batch_normalization_10_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:	?
identity??6batch_normalization_10/FusedBatchNormV3/ReadVariableOp?8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_10/ReadVariableOp?'batch_normalization_10/ReadVariableOp_1?conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_6/Conv2D:output:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
is_training( ?
leaky_re_lu_6/LeakyRelu	LeakyRelu+batch_normalization_10/FusedBatchNormV3:y:0*0
_output_shapes
:?????????  ?}
IdentityIdentity%leaky_re_lu_6/LeakyRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????  ??
NoOpNoOp7^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_1^conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@@@: : : : : 2p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_968587

inputs%
readvariableop_resource:d'
readvariableop_1_resource:d6
(fusedbatchnormv3_readvariableop_resource:d8
*fusedbatchnormv3_readvariableop_1_resource:d
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:d*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:d*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:d*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:d*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????d:d:d:d:d:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????d?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????d: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????d
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_971340

inputs%
readvariableop_resource:d'
readvariableop_1_resource:d6
(fusedbatchnormv3_readvariableop_resource:d8
*fusedbatchnormv3_readvariableop_1_resource:d
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:d*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:d*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:d*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:d*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????d:d:d:d:d:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????d?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????d: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_967461

inputs
identityP
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:?????????  ?h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:?????????  ?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_968328

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_discriminator_layer_call_fn_969104
input_1!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@$
	unknown_4:@?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?&

unknown_14:??

unknown_15:	?

unknown_16:	?

unknown_17:	?

unknown_18:	?%

unknown_19:?d

unknown_20:d

unknown_21:d

unknown_22:d

unknown_23:d$

unknown_24:d
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:?????????:?????????d*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_discriminator_layer_call_and_return_conditional_losses_969047w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????y

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
e
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_967055

inputs
identityO
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????@@@g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@@:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
-__inference_conv_block_6_layer_call_fn_967663
input_1"
unknown:@?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_6_layer_call_and_return_conditional_losses_967650x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@@@: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@@
!
_user_specified_name	input_1
?
?
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_971174

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
.__inference_sequential_10_layer_call_fn_967597
conv2d_6_input"
unknown:@?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_6_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_967569x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@@@: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????@@@
(
_user_specified_nameconv2d_6_input
?
?
-__inference_conv_block_6_layer_call_fn_967723
input_1"
unknown:@?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_6_layer_call_and_return_conditional_losses_967695x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@@@: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@@@
!
_user_specified_name	input_1
?
?
D__inference_conv2d_6_layer_call_and_return_conditional_losses_967425

inputs9
conv2d_readvariableop_resource:@?
identity??Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????@@@: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_5_layer_call_fn_970753

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_967055h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@@:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_967208
conv2d_5_input)
conv2d_5_967194:@*
batch_normalization_9_967197:@*
batch_normalization_9_967199:@*
batch_normalization_9_967201:@*
batch_normalization_9_967203:@
identity??-batch_normalization_9/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputconv2d_5_967194*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_967019?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_9_967197batch_normalization_9_967199batch_normalization_9_967201batch_normalization_9_967203*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_967040?
leaky_re_lu_5/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_967055}
IdentityIdentity&leaky_re_lu_5/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@?
NoOpNoOp.^batch_normalization_9/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????: : : : : 2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall:a ]
1
_output_shapes
:???????????
(
_user_specified_nameconv2d_5_input
?	
?
7__inference_batch_normalization_11_layer_call_fn_970946

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_967806?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_969539
input_1!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@$
	unknown_4:@?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?&

unknown_14:??

unknown_15:	?

unknown_16:	?

unknown_17:	?

unknown_18:	?%

unknown_19:?d

unknown_20:d

unknown_21:d

unknown_22:d

unknown_23:d$

unknown_24:d
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:?????????:?????????d*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_966941w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????y

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
e
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_971202

inputs
identityP
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:??????????h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
H__inference_conv_block_5_layer_call_and_return_conditional_losses_967244

inputs-
sequential_9_967232:@!
sequential_9_967234:@!
sequential_9_967236:@!
sequential_9_967238:@!
sequential_9_967240:@
identity??$sequential_9/StatefulPartitionedCall?
$sequential_9/StatefulPartitionedCallStatefulPartitionedCallinputssequential_9_967232sequential_9_967234sequential_9_967236sequential_9_967238sequential_9_967240*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_967058?
IdentityIdentity-sequential_9/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@m
NoOpNoOp%^sequential_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????: : : : : 2L
$sequential_9/StatefulPartitionedCall$sequential_9/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
-__inference_conv_block_6_layer_call_fn_969959

inputs"
unknown:@?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_6_layer_call_and_return_conditional_losses_967695x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@@@: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
-__inference_conv_block_8_layer_call_fn_970107

inputs#
unknown:??
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_8_layer_call_and_return_conditional_losses_968507x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_971156

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_12_layer_call_fn_971081

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_968181?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?

?
H__inference_conv_block_6_layer_call_and_return_conditional_losses_967650

inputs/
sequential_10_967638:@?#
sequential_10_967640:	?#
sequential_10_967642:	?#
sequential_10_967644:	?#
sequential_10_967646:	?
identity??%sequential_10/StatefulPartitionedCall?
%sequential_10/StatefulPartitionedCallStatefulPartitionedCallinputssequential_10_967638sequential_10_967640sequential_10_967642sequential_10_967644sequential_10_967646*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_10_layer_call_and_return_conditional_losses_967464?
IdentityIdentity.sequential_10/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?n
NoOpNoOp&^sequential_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@@@: : : : : 2N
%sequential_10/StatefulPartitionedCall%sequential_10/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?

?
H__inference_conv_block_8_layer_call_and_return_conditional_losses_968550
input_10
sequential_12_968538:??#
sequential_12_968540:	?#
sequential_12_968542:	?#
sequential_12_968544:	?#
sequential_12_968546:	?
identity??%sequential_12/StatefulPartitionedCall?
%sequential_12/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_12_968538sequential_12_968540sequential_12_968542sequential_12_968544sequential_12_968546*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_968276?
IdentityIdentity.sequential_12/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????n
NoOpNoOp&^sequential_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 2N
%sequential_12/StatefulPartitionedCall%sequential_12/StatefulPartitionedCall:Y U
0
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_966963

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
)__inference_conv2d_6_layer_call_fn_970765

inputs"
unknown:@?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_967425x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????@@@: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?

?
H__inference_conv_block_7_layer_call_and_return_conditional_losses_968144
input_10
sequential_11_968132:??#
sequential_11_968134:	?#
sequential_11_968136:	?#
sequential_11_968138:	?#
sequential_11_968140:	?
identity??%sequential_11/StatefulPartitionedCall?
%sequential_11/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_11_968132sequential_11_968134sequential_11_968136sequential_11_968138sequential_11_968140*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_11_layer_call_and_return_conditional_losses_967870?
IdentityIdentity.sequential_11/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????n
NoOpNoOp&^sequential_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????  ?: : : : : 2N
%sequential_11/StatefulPartitionedCall%sequential_11/StatefulPartitionedCall:Y U
0
_output_shapes
:?????????  ?
!
_user_specified_name	input_1
?
?
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_968664

inputs%
readvariableop_resource:d'
readvariableop_1_resource:d6
(fusedbatchnormv3_readvariableop_resource:d8
*fusedbatchnormv3_readvariableop_1_resource:d
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:d*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:d*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:d*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:d*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????d:d:d:d:d:*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????d?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????d: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_13_layer_call_fn_971255

inputs
unknown:d
	unknown_0:d
	unknown_1:d
	unknown_2:d
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_968664w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????d: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
I__inference_sequential_11_layer_call_and_return_conditional_losses_967975

inputs+
conv2d_7_967961:??,
batch_normalization_11_967964:	?,
batch_normalization_11_967966:	?,
batch_normalization_11_967968:	?,
batch_normalization_11_967970:	?
identity??.batch_normalization_11/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_7_967961*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_967831?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_11_967964batch_normalization_11_967966batch_normalization_11_967968batch_normalization_11_967970*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_967922?
leaky_re_lu_7/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_967867~
IdentityIdentity&leaky_re_lu_7/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp/^batch_normalization_11/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????  ?: : : : : 2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?

?
H__inference_conv_block_5_layer_call_and_return_conditional_losses_967332
input_1-
sequential_9_967320:@!
sequential_9_967322:@!
sequential_9_967324:@!
sequential_9_967326:@!
sequential_9_967328:@
identity??$sequential_9/StatefulPartitionedCall?
$sequential_9/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_9_967320sequential_9_967322sequential_9_967324sequential_9_967326sequential_9_967328*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_967058?
IdentityIdentity-sequential_9/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@m
NoOpNoOp%^sequential_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????: : : : : 2L
$sequential_9/StatefulPartitionedCall$sequential_9/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_968258

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_13_layer_call_and_return_conditional_losses_968787

inputs*
conv2d_9_968773:?d+
batch_normalization_13_968776:d+
batch_normalization_13_968778:d+
batch_normalization_13_968780:d+
batch_normalization_13_968782:d
identity??.batch_normalization_13/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_9_968773*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_968643?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0batch_normalization_13_968776batch_normalization_13_968778batch_normalization_13_968780batch_normalization_13_968782*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_968734?
leaky_re_lu_9/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_968679}
IdentityIdentity&leaky_re_lu_9/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????d?
NoOpNoOp/^batch_normalization_13/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
H__inference_conv_block_9_layer_call_and_return_conditional_losses_968956
input_1/
sequential_13_968944:?d"
sequential_13_968946:d"
sequential_13_968948:d"
sequential_13_968950:d"
sequential_13_968952:d
identity??%sequential_13/StatefulPartitionedCall?
%sequential_13/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_13_968944sequential_13_968946sequential_13_968948sequential_13_968950sequential_13_968952*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_13_layer_call_and_return_conditional_losses_968682?
IdentityIdentity.sequential_13/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????dn
NoOpNoOp&^sequential_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 2N
%sequential_13/StatefulPartitionedCall%sequential_13/StatefulPartitionedCall:Y U
0
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
I__inference_sequential_10_layer_call_and_return_conditional_losses_967631
conv2d_6_input*
conv2d_6_967617:@?,
batch_normalization_10_967620:	?,
batch_normalization_10_967622:	?,
batch_normalization_10_967624:	?,
batch_normalization_10_967626:	?
identity??.batch_normalization_10/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallconv2d_6_inputconv2d_6_967617*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_967425?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_10_967620batch_normalization_10_967622batch_normalization_10_967624batch_normalization_10_967626*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_967516?
leaky_re_lu_6/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_967461~
IdentityIdentity&leaky_re_lu_6/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ??
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@@@: : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????@@@
(
_user_specified_nameconv2d_6_input
? 
?
H__inference_sequential_9_layer_call_and_return_conditional_losses_970314

inputsA
'conv2d_5_conv2d_readvariableop_resource:@;
-batch_normalization_9_readvariableop_resource:@=
/batch_normalization_9_readvariableop_1_resource:@L
>batch_normalization_9_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:@
identity??$batch_normalization_9/AssignNewValue?&batch_normalization_9/AssignNewValue_1?5batch_normalization_9/FusedBatchNormV3/ReadVariableOp?7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_9/ReadVariableOp?&batch_normalization_9/ReadVariableOp_1?conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
$batch_normalization_9/ReadVariableOpReadVariableOp-batch_normalization_9_readvariableop_resource*
_output_shapes
:@*
dtype0?
&batch_normalization_9/ReadVariableOp_1ReadVariableOp/batch_normalization_9_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
5batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
&batch_normalization_9/FusedBatchNormV3FusedBatchNormV3conv2d_5/Conv2D:output:0,batch_normalization_9/ReadVariableOp:value:0.batch_normalization_9/ReadVariableOp_1:value:0=batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
$batch_normalization_9/AssignNewValueAssignVariableOp>batch_normalization_9_fusedbatchnormv3_readvariableop_resource3batch_normalization_9/FusedBatchNormV3:batch_mean:06^batch_normalization_9/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
&batch_normalization_9/AssignNewValue_1AssignVariableOp@batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_9/FusedBatchNormV3:batch_variance:08^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_re_lu_5/LeakyRelu	LeakyRelu*batch_normalization_9/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@@@|
IdentityIdentity%leaky_re_lu_5/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@@@?
NoOpNoOp%^batch_normalization_9/AssignNewValue'^batch_normalization_9/AssignNewValue_16^batch_normalization_9/FusedBatchNormV3/ReadVariableOp8^batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_9/ReadVariableOp'^batch_normalization_9/ReadVariableOp_1^conv2d_5/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????: : : : : 2L
$batch_normalization_9/AssignNewValue$batch_normalization_9/AssignNewValue2P
&batch_normalization_9/AssignNewValue_1&batch_normalization_9/AssignNewValue_12n
5batch_normalization_9/FusedBatchNormV3/ReadVariableOp5batch_normalization_9/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_9/FusedBatchNormV3/ReadVariableOp_17batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_9/ReadVariableOp$batch_normalization_9/ReadVariableOp2P
&batch_normalization_9/ReadVariableOp_1&batch_normalization_9/ReadVariableOp_12@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_967852

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_13_layer_call_fn_970566

inputs"
unknown:?d
	unknown_0:d
	unknown_1:d
	unknown_2:d
	unknown_3:d
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_13_layer_call_and_return_conditional_losses_968787w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_967400

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_conv2d_7_layer_call_fn_970913

inputs#
unknown:??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_967831x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????  ?: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
D__inference_conv2d_5_layer_call_and_return_conditional_losses_967019

inputs8
conv2d_readvariableop_resource:@
identity??Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_11_layer_call_and_return_conditional_losses_967870

inputs+
conv2d_7_967832:??,
batch_normalization_11_967853:	?,
batch_normalization_11_967855:	?,
batch_normalization_11_967857:	?,
batch_normalization_11_967859:	?
identity??.batch_normalization_11/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_7_967832*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_967831?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_11_967853batch_normalization_11_967855batch_normalization_11_967857batch_normalization_11_967859*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_967852?
leaky_re_lu_7/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_967867~
IdentityIdentity&leaky_re_lu_7/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp/^batch_normalization_11/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????  ?: : : : : 2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
-__inference_conv_block_6_layer_call_fn_969944

inputs"
unknown:@?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_6_layer_call_and_return_conditional_losses_967650x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@@@: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?	
?
.__inference_sequential_12_layer_call_fn_968409
conv2d_8_input#
unknown:??
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_968381x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_8_input
?	
?
7__inference_batch_normalization_13_layer_call_fn_971242

inputs
unknown:d
	unknown_0:d
	unknown_1:d
	unknown_2:d
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_968618?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????d: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????d
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_967110

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_971044

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_12_layer_call_fn_971107

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_968258x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_9_layer_call_fn_970650

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_966994?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?

?
H__inference_conv_block_8_layer_call_and_return_conditional_losses_968462

inputs0
sequential_12_968450:??#
sequential_12_968452:	?#
sequential_12_968454:	?#
sequential_12_968456:	?#
sequential_12_968458:	?
identity??%sequential_12/StatefulPartitionedCall?
%sequential_12/StatefulPartitionedCallStatefulPartitionedCallinputssequential_12_968450sequential_12_968452sequential_12_968454sequential_12_968456sequential_12_968458*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_968276?
IdentityIdentity.sequential_12/StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????n
NoOpNoOp&^sequential_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 2N
%sequential_12/StatefulPartitionedCall%sequential_12/StatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?(
?
H__inference_conv_block_8_layer_call_and_return_conditional_losses_970151

inputsQ
5sequential_12_conv2d_8_conv2d_readvariableop_resource:??K
<sequential_12_batch_normalization_12_readvariableop_resource:	?M
>sequential_12_batch_normalization_12_readvariableop_1_resource:	?\
Msequential_12_batch_normalization_12_fusedbatchnormv3_readvariableop_resource:	?^
Osequential_12_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:	?
identity??3sequential_12/batch_normalization_12/AssignNewValue?5sequential_12/batch_normalization_12/AssignNewValue_1?Dsequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp?Fsequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?3sequential_12/batch_normalization_12/ReadVariableOp?5sequential_12/batch_normalization_12/ReadVariableOp_1?,sequential_12/conv2d_8/Conv2D/ReadVariableOp?
,sequential_12/conv2d_8/Conv2D/ReadVariableOpReadVariableOp5sequential_12_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
sequential_12/conv2d_8/Conv2DConv2Dinputs4sequential_12/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
3sequential_12/batch_normalization_12/ReadVariableOpReadVariableOp<sequential_12_batch_normalization_12_readvariableop_resource*
_output_shapes	
:?*
dtype0?
5sequential_12/batch_normalization_12/ReadVariableOp_1ReadVariableOp>sequential_12_batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Dsequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_12_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Fsequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_12_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
5sequential_12/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3&sequential_12/conv2d_8/Conv2D:output:0;sequential_12/batch_normalization_12/ReadVariableOp:value:0=sequential_12/batch_normalization_12/ReadVariableOp_1:value:0Lsequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
3sequential_12/batch_normalization_12/AssignNewValueAssignVariableOpMsequential_12_batch_normalization_12_fusedbatchnormv3_readvariableop_resourceBsequential_12/batch_normalization_12/FusedBatchNormV3:batch_mean:0E^sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
5sequential_12/batch_normalization_12/AssignNewValue_1AssignVariableOpOsequential_12_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resourceFsequential_12/batch_normalization_12/FusedBatchNormV3:batch_variance:0G^sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
%sequential_12/leaky_re_lu_8/LeakyRelu	LeakyRelu9sequential_12/batch_normalization_12/FusedBatchNormV3:y:0*0
_output_shapes
:???????????
IdentityIdentity3sequential_12/leaky_re_lu_8/LeakyRelu:activations:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp4^sequential_12/batch_normalization_12/AssignNewValue6^sequential_12/batch_normalization_12/AssignNewValue_1E^sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOpG^sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_14^sequential_12/batch_normalization_12/ReadVariableOp6^sequential_12/batch_normalization_12/ReadVariableOp_1-^sequential_12/conv2d_8/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 2j
3sequential_12/batch_normalization_12/AssignNewValue3sequential_12/batch_normalization_12/AssignNewValue2n
5sequential_12/batch_normalization_12/AssignNewValue_15sequential_12/batch_normalization_12/AssignNewValue_12?
Dsequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOpDsequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp2?
Fsequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1Fsequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12j
3sequential_12/batch_normalization_12/ReadVariableOp3sequential_12/batch_normalization_12/ReadVariableOp2n
5sequential_12/batch_normalization_12/ReadVariableOp_15sequential_12/batch_normalization_12/ReadVariableOp_12\
,sequential_12/conv2d_8/Conv2D/ReadVariableOp,sequential_12/conv2d_8/Conv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_970896

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????  ??
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????  ?: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?

?
H__inference_conv_block_5_layer_call_and_return_conditional_losses_967347
input_1-
sequential_9_967335:@!
sequential_9_967337:@!
sequential_9_967339:@!
sequential_9_967341:@!
sequential_9_967343:@
identity??$sequential_9/StatefulPartitionedCall?
$sequential_9/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_9_967335sequential_9_967337sequential_9_967339sequential_9_967341sequential_9_967343*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_967163?
IdentityIdentity-sequential_9/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@m
NoOpNoOp%^sequential_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????: : : : : 2L
$sequential_9/StatefulPartitionedCall$sequential_9/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
I__inference_sequential_13_layer_call_and_return_conditional_losses_968849
conv2d_9_input*
conv2d_9_968835:?d+
batch_normalization_13_968838:d+
batch_normalization_13_968840:d+
batch_normalization_13_968842:d+
batch_normalization_13_968844:d
identity??.batch_normalization_13/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallconv2d_9_inputconv2d_9_968835*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_968643?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0batch_normalization_13_968838batch_normalization_13_968840batch_normalization_13_968842batch_normalization_13_968844*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_968734?
leaky_re_lu_9/PartitionedCallPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_968679}
IdentityIdentity&leaky_re_lu_9/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????d?
NoOpNoOp/^batch_normalization_13/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:` \
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_9_input
?
?
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_971138

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
-__inference_conv_block_5_layer_call_fn_969885

inputs!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_5_layer_call_and_return_conditional_losses_967289w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
H__inference_conv_block_9_layer_call_and_return_conditional_losses_970203

inputsP
5sequential_13_conv2d_9_conv2d_readvariableop_resource:?dJ
<sequential_13_batch_normalization_13_readvariableop_resource:dL
>sequential_13_batch_normalization_13_readvariableop_1_resource:d[
Msequential_13_batch_normalization_13_fusedbatchnormv3_readvariableop_resource:d]
Osequential_13_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:d
identity??Dsequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp?Fsequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?3sequential_13/batch_normalization_13/ReadVariableOp?5sequential_13/batch_normalization_13/ReadVariableOp_1?,sequential_13/conv2d_9/Conv2D/ReadVariableOp?
,sequential_13/conv2d_9/Conv2D/ReadVariableOpReadVariableOp5sequential_13_conv2d_9_conv2d_readvariableop_resource*'
_output_shapes
:?d*
dtype0?
sequential_13/conv2d_9/Conv2DConv2Dinputs4sequential_13/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d*
paddingSAME*
strides
?
3sequential_13/batch_normalization_13/ReadVariableOpReadVariableOp<sequential_13_batch_normalization_13_readvariableop_resource*
_output_shapes
:d*
dtype0?
5sequential_13/batch_normalization_13/ReadVariableOp_1ReadVariableOp>sequential_13_batch_normalization_13_readvariableop_1_resource*
_output_shapes
:d*
dtype0?
Dsequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_13_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:d*
dtype0?
Fsequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_13_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:d*
dtype0?
5sequential_13/batch_normalization_13/FusedBatchNormV3FusedBatchNormV3&sequential_13/conv2d_9/Conv2D:output:0;sequential_13/batch_normalization_13/ReadVariableOp:value:0=sequential_13/batch_normalization_13/ReadVariableOp_1:value:0Lsequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????d:d:d:d:d:*
epsilon%o?:*
is_training( ?
%sequential_13/leaky_re_lu_9/LeakyRelu	LeakyRelu9sequential_13/batch_normalization_13/FusedBatchNormV3:y:0*/
_output_shapes
:?????????d?
IdentityIdentity3sequential_13/leaky_re_lu_9/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????d?
NoOpNoOpE^sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOpG^sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_14^sequential_13/batch_normalization_13/ReadVariableOp6^sequential_13/batch_normalization_13/ReadVariableOp_1-^sequential_13/conv2d_9/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 2?
Dsequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOpDsequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp2?
Fsequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1Fsequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12j
3sequential_13/batch_normalization_13/ReadVariableOp3sequential_13/batch_normalization_13/ReadVariableOp2n
5sequential_13/batch_normalization_13/ReadVariableOp_15sequential_13/batch_normalization_13/ReadVariableOp_12\
,sequential_13/conv2d_9/Conv2D/ReadVariableOp,sequential_13/conv2d_9/Conv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
7__inference_batch_normalization_11_layer_call_fn_970972

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_967922x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_970860

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
.__inference_sequential_11_layer_call_fn_967883
conv2d_7_input#
unknown:??
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_7_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_11_layer_call_and_return_conditional_losses_967870x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????  ?: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
0
_output_shapes
:?????????  ?
(
_user_specified_nameconv2d_7_input
ʫ
?#
I__inference_discriminator_layer_call_and_return_conditional_losses_969756

inputs[
Aconv_block_5_sequential_9_conv2d_5_conv2d_readvariableop_resource:@U
Gconv_block_5_sequential_9_batch_normalization_9_readvariableop_resource:@W
Iconv_block_5_sequential_9_batch_normalization_9_readvariableop_1_resource:@f
Xconv_block_5_sequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:@h
Zconv_block_5_sequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:@]
Bconv_block_6_sequential_10_conv2d_6_conv2d_readvariableop_resource:@?X
Iconv_block_6_sequential_10_batch_normalization_10_readvariableop_resource:	?Z
Kconv_block_6_sequential_10_batch_normalization_10_readvariableop_1_resource:	?i
Zconv_block_6_sequential_10_batch_normalization_10_fusedbatchnormv3_readvariableop_resource:	?k
\conv_block_6_sequential_10_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:	?^
Bconv_block_7_sequential_11_conv2d_7_conv2d_readvariableop_resource:??X
Iconv_block_7_sequential_11_batch_normalization_11_readvariableop_resource:	?Z
Kconv_block_7_sequential_11_batch_normalization_11_readvariableop_1_resource:	?i
Zconv_block_7_sequential_11_batch_normalization_11_fusedbatchnormv3_readvariableop_resource:	?k
\conv_block_7_sequential_11_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:	?^
Bconv_block_8_sequential_12_conv2d_8_conv2d_readvariableop_resource:??X
Iconv_block_8_sequential_12_batch_normalization_12_readvariableop_resource:	?Z
Kconv_block_8_sequential_12_batch_normalization_12_readvariableop_1_resource:	?i
Zconv_block_8_sequential_12_batch_normalization_12_fusedbatchnormv3_readvariableop_resource:	?k
\conv_block_8_sequential_12_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:	?]
Bconv_block_9_sequential_13_conv2d_9_conv2d_readvariableop_resource:?dW
Iconv_block_9_sequential_13_batch_normalization_13_readvariableop_resource:dY
Kconv_block_9_sequential_13_batch_normalization_13_readvariableop_1_resource:dh
Zconv_block_9_sequential_13_batch_normalization_13_fusedbatchnormv3_readvariableop_resource:dj
\conv_block_9_sequential_13_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:dB
(conv2d_10_conv2d_readvariableop_resource:d
identity

identity_1??conv2d_10/Conv2D/ReadVariableOp?Oconv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp?Qconv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?>conv_block_5/sequential_9/batch_normalization_9/ReadVariableOp?@conv_block_5/sequential_9/batch_normalization_9/ReadVariableOp_1?8conv_block_5/sequential_9/conv2d_5/Conv2D/ReadVariableOp?Qconv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp?Sconv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1?@conv_block_6/sequential_10/batch_normalization_10/ReadVariableOp?Bconv_block_6/sequential_10/batch_normalization_10/ReadVariableOp_1?9conv_block_6/sequential_10/conv2d_6/Conv2D/ReadVariableOp?Qconv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp?Sconv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1?@conv_block_7/sequential_11/batch_normalization_11/ReadVariableOp?Bconv_block_7/sequential_11/batch_normalization_11/ReadVariableOp_1?9conv_block_7/sequential_11/conv2d_7/Conv2D/ReadVariableOp?Qconv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp?Sconv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?@conv_block_8/sequential_12/batch_normalization_12/ReadVariableOp?Bconv_block_8/sequential_12/batch_normalization_12/ReadVariableOp_1?9conv_block_8/sequential_12/conv2d_8/Conv2D/ReadVariableOp?Qconv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp?Sconv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1?@conv_block_9/sequential_13/batch_normalization_13/ReadVariableOp?Bconv_block_9/sequential_13/batch_normalization_13/ReadVariableOp_1?9conv_block_9/sequential_13/conv2d_9/Conv2D/ReadVariableOp?
8conv_block_5/sequential_9/conv2d_5/Conv2D/ReadVariableOpReadVariableOpAconv_block_5_sequential_9_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
)conv_block_5/sequential_9/conv2d_5/Conv2DConv2Dinputs@conv_block_5/sequential_9/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
>conv_block_5/sequential_9/batch_normalization_9/ReadVariableOpReadVariableOpGconv_block_5_sequential_9_batch_normalization_9_readvariableop_resource*
_output_shapes
:@*
dtype0?
@conv_block_5/sequential_9/batch_normalization_9/ReadVariableOp_1ReadVariableOpIconv_block_5_sequential_9_batch_normalization_9_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
Oconv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpXconv_block_5_sequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
Qconv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpZconv_block_5_sequential_9_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
@conv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3FusedBatchNormV32conv_block_5/sequential_9/conv2d_5/Conv2D:output:0Fconv_block_5/sequential_9/batch_normalization_9/ReadVariableOp:value:0Hconv_block_5/sequential_9/batch_normalization_9/ReadVariableOp_1:value:0Wconv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Yconv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
is_training( ?
1conv_block_5/sequential_9/leaky_re_lu_5/LeakyRelu	LeakyReluDconv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3:y:0*/
_output_shapes
:?????????@@@?
9conv_block_6/sequential_10/conv2d_6/Conv2D/ReadVariableOpReadVariableOpBconv_block_6_sequential_10_conv2d_6_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
*conv_block_6/sequential_10/conv2d_6/Conv2DConv2D?conv_block_5/sequential_9/leaky_re_lu_5/LeakyRelu:activations:0Aconv_block_6/sequential_10/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
@conv_block_6/sequential_10/batch_normalization_10/ReadVariableOpReadVariableOpIconv_block_6_sequential_10_batch_normalization_10_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Bconv_block_6/sequential_10/batch_normalization_10/ReadVariableOp_1ReadVariableOpKconv_block_6_sequential_10_batch_normalization_10_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Qconv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpZconv_block_6_sequential_10_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Sconv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp\conv_block_6_sequential_10_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Bconv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3FusedBatchNormV33conv_block_6/sequential_10/conv2d_6/Conv2D:output:0Hconv_block_6/sequential_10/batch_normalization_10/ReadVariableOp:value:0Jconv_block_6/sequential_10/batch_normalization_10/ReadVariableOp_1:value:0Yconv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0[conv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
is_training( ?
2conv_block_6/sequential_10/leaky_re_lu_6/LeakyRelu	LeakyReluFconv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3:y:0*0
_output_shapes
:?????????  ??
9conv_block_7/sequential_11/conv2d_7/Conv2D/ReadVariableOpReadVariableOpBconv_block_7_sequential_11_conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
*conv_block_7/sequential_11/conv2d_7/Conv2DConv2D@conv_block_6/sequential_10/leaky_re_lu_6/LeakyRelu:activations:0Aconv_block_7/sequential_11/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
@conv_block_7/sequential_11/batch_normalization_11/ReadVariableOpReadVariableOpIconv_block_7_sequential_11_batch_normalization_11_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Bconv_block_7/sequential_11/batch_normalization_11/ReadVariableOp_1ReadVariableOpKconv_block_7_sequential_11_batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Qconv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpZconv_block_7_sequential_11_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Sconv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp\conv_block_7_sequential_11_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Bconv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3FusedBatchNormV33conv_block_7/sequential_11/conv2d_7/Conv2D:output:0Hconv_block_7/sequential_11/batch_normalization_11/ReadVariableOp:value:0Jconv_block_7/sequential_11/batch_normalization_11/ReadVariableOp_1:value:0Yconv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0[conv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( ?
2conv_block_7/sequential_11/leaky_re_lu_7/LeakyRelu	LeakyReluFconv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3:y:0*0
_output_shapes
:???????????
9conv_block_8/sequential_12/conv2d_8/Conv2D/ReadVariableOpReadVariableOpBconv_block_8_sequential_12_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
*conv_block_8/sequential_12/conv2d_8/Conv2DConv2D@conv_block_7/sequential_11/leaky_re_lu_7/LeakyRelu:activations:0Aconv_block_8/sequential_12/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
@conv_block_8/sequential_12/batch_normalization_12/ReadVariableOpReadVariableOpIconv_block_8_sequential_12_batch_normalization_12_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Bconv_block_8/sequential_12/batch_normalization_12/ReadVariableOp_1ReadVariableOpKconv_block_8_sequential_12_batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Qconv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpZconv_block_8_sequential_12_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Sconv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp\conv_block_8_sequential_12_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Bconv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3FusedBatchNormV33conv_block_8/sequential_12/conv2d_8/Conv2D:output:0Hconv_block_8/sequential_12/batch_normalization_12/ReadVariableOp:value:0Jconv_block_8/sequential_12/batch_normalization_12/ReadVariableOp_1:value:0Yconv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0[conv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( ?
2conv_block_8/sequential_12/leaky_re_lu_8/LeakyRelu	LeakyReluFconv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3:y:0*0
_output_shapes
:???????????
9conv_block_9/sequential_13/conv2d_9/Conv2D/ReadVariableOpReadVariableOpBconv_block_9_sequential_13_conv2d_9_conv2d_readvariableop_resource*'
_output_shapes
:?d*
dtype0?
*conv_block_9/sequential_13/conv2d_9/Conv2DConv2D@conv_block_8/sequential_12/leaky_re_lu_8/LeakyRelu:activations:0Aconv_block_9/sequential_13/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????d*
paddingSAME*
strides
?
@conv_block_9/sequential_13/batch_normalization_13/ReadVariableOpReadVariableOpIconv_block_9_sequential_13_batch_normalization_13_readvariableop_resource*
_output_shapes
:d*
dtype0?
Bconv_block_9/sequential_13/batch_normalization_13/ReadVariableOp_1ReadVariableOpKconv_block_9_sequential_13_batch_normalization_13_readvariableop_1_resource*
_output_shapes
:d*
dtype0?
Qconv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpZconv_block_9_sequential_13_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:d*
dtype0?
Sconv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp\conv_block_9_sequential_13_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:d*
dtype0?
Bconv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3FusedBatchNormV33conv_block_9/sequential_13/conv2d_9/Conv2D:output:0Hconv_block_9/sequential_13/batch_normalization_13/ReadVariableOp:value:0Jconv_block_9/sequential_13/batch_normalization_13/ReadVariableOp_1:value:0Yconv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0[conv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????d:d:d:d:d:*
epsilon%o?:*
is_training( ?
2conv_block_9/sequential_13/leaky_re_lu_9/LeakyRelu	LeakyReluFconv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3:y:0*/
_output_shapes
:?????????d?
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:d*
dtype0?
conv2d_10/Conv2DConv2D@conv_block_9/sequential_13/leaky_re_lu_9/LeakyRelu:activations:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
q
conv2d_10/SigmoidSigmoidconv2d_10/Conv2D:output:0*
T0*/
_output_shapes
:?????????l
IdentityIdentityconv2d_10/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:??????????

Identity_1Identity@conv_block_9/sequential_13/leaky_re_lu_9/LeakyRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????d?
NoOpNoOp ^conv2d_10/Conv2D/ReadVariableOpP^conv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpR^conv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?^conv_block_5/sequential_9/batch_normalization_9/ReadVariableOpA^conv_block_5/sequential_9/batch_normalization_9/ReadVariableOp_19^conv_block_5/sequential_9/conv2d_5/Conv2D/ReadVariableOpR^conv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOpT^conv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1A^conv_block_6/sequential_10/batch_normalization_10/ReadVariableOpC^conv_block_6/sequential_10/batch_normalization_10/ReadVariableOp_1:^conv_block_6/sequential_10/conv2d_6/Conv2D/ReadVariableOpR^conv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOpT^conv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1A^conv_block_7/sequential_11/batch_normalization_11/ReadVariableOpC^conv_block_7/sequential_11/batch_normalization_11/ReadVariableOp_1:^conv_block_7/sequential_11/conv2d_7/Conv2D/ReadVariableOpR^conv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOpT^conv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1A^conv_block_8/sequential_12/batch_normalization_12/ReadVariableOpC^conv_block_8/sequential_12/batch_normalization_12/ReadVariableOp_1:^conv_block_8/sequential_12/conv2d_8/Conv2D/ReadVariableOpR^conv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOpT^conv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1A^conv_block_9/sequential_13/batch_normalization_13/ReadVariableOpC^conv_block_9/sequential_13/batch_normalization_13/ReadVariableOp_1:^conv_block_9/sequential_13/conv2d_9/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2?
Oconv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOpOconv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2?
Qconv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1Qconv_block_5/sequential_9/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12?
>conv_block_5/sequential_9/batch_normalization_9/ReadVariableOp>conv_block_5/sequential_9/batch_normalization_9/ReadVariableOp2?
@conv_block_5/sequential_9/batch_normalization_9/ReadVariableOp_1@conv_block_5/sequential_9/batch_normalization_9/ReadVariableOp_12t
8conv_block_5/sequential_9/conv2d_5/Conv2D/ReadVariableOp8conv_block_5/sequential_9/conv2d_5/Conv2D/ReadVariableOp2?
Qconv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOpQconv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp2?
Sconv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1Sconv_block_6/sequential_10/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12?
@conv_block_6/sequential_10/batch_normalization_10/ReadVariableOp@conv_block_6/sequential_10/batch_normalization_10/ReadVariableOp2?
Bconv_block_6/sequential_10/batch_normalization_10/ReadVariableOp_1Bconv_block_6/sequential_10/batch_normalization_10/ReadVariableOp_12v
9conv_block_6/sequential_10/conv2d_6/Conv2D/ReadVariableOp9conv_block_6/sequential_10/conv2d_6/Conv2D/ReadVariableOp2?
Qconv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOpQconv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp2?
Sconv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1Sconv_block_7/sequential_11/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12?
@conv_block_7/sequential_11/batch_normalization_11/ReadVariableOp@conv_block_7/sequential_11/batch_normalization_11/ReadVariableOp2?
Bconv_block_7/sequential_11/batch_normalization_11/ReadVariableOp_1Bconv_block_7/sequential_11/batch_normalization_11/ReadVariableOp_12v
9conv_block_7/sequential_11/conv2d_7/Conv2D/ReadVariableOp9conv_block_7/sequential_11/conv2d_7/Conv2D/ReadVariableOp2?
Qconv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOpQconv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp2?
Sconv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1Sconv_block_8/sequential_12/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12?
@conv_block_8/sequential_12/batch_normalization_12/ReadVariableOp@conv_block_8/sequential_12/batch_normalization_12/ReadVariableOp2?
Bconv_block_8/sequential_12/batch_normalization_12/ReadVariableOp_1Bconv_block_8/sequential_12/batch_normalization_12/ReadVariableOp_12v
9conv_block_8/sequential_12/conv2d_8/Conv2D/ReadVariableOp9conv_block_8/sequential_12/conv2d_8/Conv2D/ReadVariableOp2?
Qconv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOpQconv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp2?
Sconv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1Sconv_block_9/sequential_13/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12?
@conv_block_9/sequential_13/batch_normalization_13/ReadVariableOp@conv_block_9/sequential_13/batch_normalization_13/ReadVariableOp2?
Bconv_block_9/sequential_13/batch_normalization_13/ReadVariableOp_1Bconv_block_9/sequential_13/batch_normalization_13/ReadVariableOp_12v
9conv_block_9/sequential_13/conv2d_9/Conv2D/ReadVariableOp9conv_block_9/sequential_13/conv2d_9/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?!
?
I__inference_sequential_12_layer_call_and_return_conditional_losses_970536

inputsC
'conv2d_8_conv2d_readvariableop_resource:??=
.batch_normalization_12_readvariableop_resource:	??
0batch_normalization_12_readvariableop_1_resource:	?N
?batch_normalization_12_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:	?
identity??%batch_normalization_12/AssignNewValue?'batch_normalization_12/AssignNewValue_1?6batch_normalization_12/FusedBatchNormV3/ReadVariableOp?8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_12/ReadVariableOp?'batch_normalization_12/ReadVariableOp_1?conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_8/Conv2DConv2Dinputs&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
%batch_normalization_12/ReadVariableOpReadVariableOp.batch_normalization_12_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_12/ReadVariableOp_1ReadVariableOp0batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3conv2d_8/Conv2D:output:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_12/AssignNewValueAssignVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource4batch_normalization_12/FusedBatchNormV3:batch_mean:07^batch_normalization_12/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization_12/AssignNewValue_1AssignVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_12/FusedBatchNormV3:batch_variance:09^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
leaky_re_lu_8/LeakyRelu	LeakyRelu+batch_normalization_12/FusedBatchNormV3:y:0*0
_output_shapes
:??????????}
IdentityIdentity%leaky_re_lu_8/LeakyRelu:activations:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp&^batch_normalization_12/AssignNewValue(^batch_normalization_12/AssignNewValue_17^batch_normalization_12/FusedBatchNormV3/ReadVariableOp9^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_12/ReadVariableOp(^batch_normalization_12/ReadVariableOp_1^conv2d_8/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 2N
%batch_normalization_12/AssignNewValue%batch_normalization_12/AssignNewValue2R
'batch_normalization_12/AssignNewValue_1'batch_normalization_12/AssignNewValue_12p
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp6batch_normalization_12/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_18batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_12/ReadVariableOp%batch_normalization_12/ReadVariableOp2R
'batch_normalization_12/ReadVariableOp_1'batch_normalization_12/ReadVariableOp_12@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_970842

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_10_layer_call_and_return_conditional_losses_967569

inputs*
conv2d_6_967555:@?,
batch_normalization_10_967558:	?,
batch_normalization_10_967560:	?,
batch_normalization_10_967562:	?,
batch_normalization_10_967564:	?
identity??.batch_normalization_10/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_6_967555*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_967425?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_10_967558batch_normalization_10_967560batch_normalization_10_967562batch_normalization_10_967564*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_967516?
leaky_re_lu_6/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_967461~
IdentityIdentity&leaky_re_lu_6/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ??
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@@@: : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
.__inference_sequential_12_layer_call_fn_970492

inputs#
unknown:??
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_968381x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_11_layer_call_and_return_conditional_losses_970440

inputsC
'conv2d_7_conv2d_readvariableop_resource:??=
.batch_normalization_11_readvariableop_resource:	??
0batch_normalization_11_readvariableop_1_resource:	?N
?batch_normalization_11_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:	?
identity??6batch_normalization_11/FusedBatchNormV3/ReadVariableOp?8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_11/ReadVariableOp?'batch_normalization_11/ReadVariableOp_1?conv2d_7/Conv2D/ReadVariableOp?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_7/Conv2DConv2Dinputs&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_7/Conv2D:output:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( ?
leaky_re_lu_7/LeakyRelu	LeakyRelu+batch_normalization_11/FusedBatchNormV3:y:0*0
_output_shapes
:??????????}
IdentityIdentity%leaky_re_lu_7/LeakyRelu:activations:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp7^batch_normalization_11/FusedBatchNormV3/ReadVariableOp9^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_11/ReadVariableOp(^batch_normalization_11/ReadVariableOp_1^conv2d_7/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????  ?: : : : : 2p
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp6batch_normalization_11/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_18batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_11/ReadVariableOp%batch_normalization_11/ReadVariableOp2R
'batch_normalization_11/ReadVariableOp_1'batch_normalization_11/ReadVariableOp_12@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
I__inference_sequential_10_layer_call_and_return_conditional_losses_967464

inputs*
conv2d_6_967426:@?,
batch_normalization_10_967447:	?,
batch_normalization_10_967449:	?,
batch_normalization_10_967451:	?,
batch_normalization_10_967453:	?
identity??.batch_normalization_10/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_6_967426*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_967425?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_10_967447batch_normalization_10_967449batch_normalization_10_967451batch_normalization_10_967453*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_967446?
leaky_re_lu_6/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_967461~
IdentityIdentity&leaky_re_lu_6/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ??
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????@@@: : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?	
?
7__inference_batch_normalization_11_layer_call_fn_970933

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_967775?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_970730

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_968181

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_971304

inputs%
readvariableop_resource:d'
readvariableop_1_resource:d6
(fusedbatchnormv3_readvariableop_resource:d8
*fusedbatchnormv3_readvariableop_1_resource:d
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:d*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:d*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:d*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:d*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????d:d:d:d:d:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????d?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????d: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????d
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_970748

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_970878

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
is_training( l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????  ??
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????  ?: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_6_layer_call_fn_970901

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_967461i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????  ?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_967867

inputs
identityP
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:??????????h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_12_layer_call_and_return_conditional_losses_968426
conv2d_8_input+
conv2d_8_968412:??,
batch_normalization_12_968415:	?,
batch_normalization_12_968417:	?,
batch_normalization_12_968419:	?,
batch_normalization_12_968421:	?
identity??.batch_normalization_12/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputconv2d_8_968412*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_968237?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_12_968415batch_normalization_12_968417batch_normalization_12_968419batch_normalization_12_968421*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_968258?
leaky_re_lu_8/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_968273~
IdentityIdentity&leaky_re_lu_8/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp/^batch_normalization_12/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:` \
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_8_input
?
?
I__inference_sequential_12_layer_call_and_return_conditional_losses_968276

inputs+
conv2d_8_968238:??,
batch_normalization_12_968259:	?,
batch_normalization_12_968261:	?,
batch_normalization_12_968263:	?,
batch_normalization_12_968265:	?
identity??.batch_normalization_12/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8_968238*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_968237?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_12_968259batch_normalization_12_968261batch_normalization_12_968263batch_normalization_12_968265*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_968258?
leaky_re_lu_8/PartitionedCallPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_968273~
IdentityIdentity&leaky_re_lu_8/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp/^batch_normalization_12/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????: : : : : 2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
-__inference_conv_block_7_layer_call_fn_970033

inputs#
unknown:??
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_7_layer_call_and_return_conditional_losses_968101x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????  ?: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?+
?

I__inference_discriminator_layer_call_and_return_conditional_losses_969047

inputs-
conv_block_5_968978:@!
conv_block_5_968980:@!
conv_block_5_968982:@!
conv_block_5_968984:@!
conv_block_5_968986:@.
conv_block_6_968989:@?"
conv_block_6_968991:	?"
conv_block_6_968993:	?"
conv_block_6_968995:	?"
conv_block_6_968997:	?/
conv_block_7_969000:??"
conv_block_7_969002:	?"
conv_block_7_969004:	?"
conv_block_7_969006:	?"
conv_block_7_969008:	?/
conv_block_8_969011:??"
conv_block_8_969013:	?"
conv_block_8_969015:	?"
conv_block_8_969017:	?"
conv_block_8_969019:	?.
conv_block_9_969022:?d!
conv_block_9_969024:d!
conv_block_9_969026:d!
conv_block_9_969028:d!
conv_block_9_969030:d*
conv2d_10_969042:d
identity

identity_1??!conv2d_10/StatefulPartitionedCall?$conv_block_5/StatefulPartitionedCall?$conv_block_6/StatefulPartitionedCall?$conv_block_7/StatefulPartitionedCall?$conv_block_8/StatefulPartitionedCall?$conv_block_9/StatefulPartitionedCall?
$conv_block_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv_block_5_968978conv_block_5_968980conv_block_5_968982conv_block_5_968984conv_block_5_968986*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_5_layer_call_and_return_conditional_losses_967244?
$conv_block_6/StatefulPartitionedCallStatefulPartitionedCall-conv_block_5/StatefulPartitionedCall:output:0conv_block_6_968989conv_block_6_968991conv_block_6_968993conv_block_6_968995conv_block_6_968997*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_6_layer_call_and_return_conditional_losses_967650?
$conv_block_7/StatefulPartitionedCallStatefulPartitionedCall-conv_block_6/StatefulPartitionedCall:output:0conv_block_7_969000conv_block_7_969002conv_block_7_969004conv_block_7_969006conv_block_7_969008*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_7_layer_call_and_return_conditional_losses_968056?
$conv_block_8/StatefulPartitionedCallStatefulPartitionedCall-conv_block_7/StatefulPartitionedCall:output:0conv_block_8_969011conv_block_8_969013conv_block_8_969015conv_block_8_969017conv_block_8_969019*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_8_layer_call_and_return_conditional_losses_968462?
$conv_block_9/StatefulPartitionedCallStatefulPartitionedCall-conv_block_8/StatefulPartitionedCall:output:0conv_block_9_969022conv_block_9_969024conv_block_9_969026conv_block_9_969028conv_block_9_969030*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_conv_block_9_layer_call_and_return_conditional_losses_968868?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall-conv_block_9/StatefulPartitionedCall:output:0conv2d_10_969042*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_969041?
IdentityIdentity*conv2d_10/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????

Identity_1Identity-conv_block_9/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????d?
NoOpNoOp"^conv2d_10/StatefulPartitionedCall%^conv_block_5/StatefulPartitionedCall%^conv_block_6/StatefulPartitionedCall%^conv_block_7/StatefulPartitionedCall%^conv_block_8/StatefulPartitionedCall%^conv_block_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2L
$conv_block_5/StatefulPartitionedCall$conv_block_5/StatefulPartitionedCall2L
$conv_block_6/StatefulPartitionedCall$conv_block_6/StatefulPartitionedCall2L
$conv_block_7/StatefulPartitionedCall$conv_block_7/StatefulPartitionedCall2L
$conv_block_8/StatefulPartitionedCall$conv_block_8/StatefulPartitionedCall2L
$conv_block_9/StatefulPartitionedCall$conv_block_9/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_1:
serving_default_input_1:0???????????D
output_18
StatefulPartitionedCall:0?????????D
output_28
StatefulPartitionedCall:1?????????dtensorflow/serving/predict:??
?
	encoder_1
	encoder_2
	encoder_3
	encoder_4

center
outputs
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_model
?

conv_layer
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_model
?

conv_layer
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_model
?

conv_layer
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_model
?

conv_layer
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_model
?
 
conv_layer
!	variables
"trainable_variables
#regularization_losses
$	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_model
?

%kernel
&	variables
'trainable_variables
(regularization_losses
)	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
*0
+1
,2
-3
.4
/5
06
17
28
39
410
511
612
713
814
915
:16
;17
<18
=19
>20
?21
@22
A23
B24
%25"
trackable_list_wrapper
?
*0
+1
,2
/3
04
15
46
57
68
99
:10
;11
>12
?13
@14
%15"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
	regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?
Hlayer_with_weights-0
Hlayer-0
Ilayer_with_weights-1
Ilayer-1
Jlayer-2
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
C
*0
+1
,2
-3
.4"
trackable_list_wrapper
5
*0
+1
,2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
Tlayer_with_weights-0
Tlayer-0
Ulayer_with_weights-1
Ulayer-1
Vlayer-2
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
C
/0
01
12
23
34"
trackable_list_wrapper
5
/0
01
12"
trackable_list_wrapper
 "
trackable_list_wrapper
?
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
`layer_with_weights-0
`layer-0
alayer_with_weights-1
alayer-1
blayer-2
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
C
40
51
62
73
84"
trackable_list_wrapper
5
40
51
62"
trackable_list_wrapper
 "
trackable_list_wrapper
?
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
llayer_with_weights-0
llayer-0
mlayer_with_weights-1
mlayer-1
nlayer-2
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
C
90
:1
;2
<3
=4"
trackable_list_wrapper
5
90
:1
;2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
xlayer_with_weights-0
xlayer-0
ylayer_with_weights-1
ylayer-1
zlayer-2
{	variables
|trainable_variables
}regularization_losses
~	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
C
>0
?1
@2
A3
B4"
trackable_list_wrapper
5
>0
?1
@2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
!	variables
"trainable_variables
#regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
8:6d2discriminator/conv2d_10/kernel
'
%0"
trackable_list_wrapper
'
%0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
&	variables
'trainable_variables
(regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'@2conv2d_5/kernel
):'@2batch_normalization_9/gamma
(:&@2batch_normalization_9/beta
1:/@ (2!batch_normalization_9/moving_mean
5:3@ (2%batch_normalization_9/moving_variance
*:(@?2conv2d_6/kernel
+:)?2batch_normalization_10/gamma
*:(?2batch_normalization_10/beta
3:1? (2"batch_normalization_10/moving_mean
7:5? (2&batch_normalization_10/moving_variance
+:)??2conv2d_7/kernel
+:)?2batch_normalization_11/gamma
*:(?2batch_normalization_11/beta
3:1? (2"batch_normalization_11/moving_mean
7:5? (2&batch_normalization_11/moving_variance
+:)??2conv2d_8/kernel
+:)?2batch_normalization_12/gamma
*:(?2batch_normalization_12/beta
3:1? (2"batch_normalization_12/moving_mean
7:5? (2&batch_normalization_12/moving_variance
*:(?d2conv2d_9/kernel
*:(d2batch_normalization_13/gamma
):'d2batch_normalization_13/beta
2:0d (2"batch_normalization_13/moving_mean
6:4d (2&batch_normalization_13/moving_variance
f
-0
.1
22
33
74
85
<6
=7
A8
B9"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

*kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	+gamma
,beta
-moving_mean
.moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
C
*0
+1
,2
-3
.4"
trackable_list_wrapper
5
*0
+1
,2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
-0
.1"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

/kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	0gamma
1beta
2moving_mean
3moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
C
/0
01
12
23
34"
trackable_list_wrapper
5
/0
01
12"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
20
31"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

4kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	5gamma
6beta
7moving_mean
8moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
C
40
51
62
73
84"
trackable_list_wrapper
5
40
51
62"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
c	variables
dtrainable_variables
eregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
70
81"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

9kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	:gamma
;beta
<moving_mean
=moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
C
90
:1
;2
<3
=4"
trackable_list_wrapper
5
90
:1
;2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
o	variables
ptrainable_variables
qregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
<0
=1"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

>kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	?gamma
@beta
Amoving_mean
Bmoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
C
>0
?1
@2
A3
B4"
trackable_list_wrapper
5
>0
?1
@2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
{	variables
|trainable_variables
}regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
A0
B1"
trackable_list_wrapper
'
 0"
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
'
*0"
trackable_list_wrapper
'
*0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
+0
,1
-2
.3"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
-0
.1"
trackable_list_wrapper
5
H0
I1
J2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
/0"
trackable_list_wrapper
'
/0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
00
11
22
33"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
20
31"
trackable_list_wrapper
5
T0
U1
V2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
40"
trackable_list_wrapper
'
40"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
50
61
72
83"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
70
81"
trackable_list_wrapper
5
`0
a1
b2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
90"
trackable_list_wrapper
'
90"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
:0
;1
<2
=3"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
<0
=1"
trackable_list_wrapper
5
l0
m1
n2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
>0"
trackable_list_wrapper
'
>0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
?0
@1
A2
B3"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
A0
B1"
trackable_list_wrapper
5
x0
y1
z2"
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
-0
.1"
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
20
31"
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
70
81"
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
<0
=1"
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
A0
B1"
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
?2?
.__inference_discriminator_layer_call_fn_969104
.__inference_discriminator_layer_call_fn_969598
.__inference_discriminator_layer_call_fn_969657
.__inference_discriminator_layer_call_fn_969352?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_discriminator_layer_call_and_return_conditional_losses_969756
I__inference_discriminator_layer_call_and_return_conditional_losses_969855
I__inference_discriminator_layer_call_and_return_conditional_losses_969415
I__inference_discriminator_layer_call_and_return_conditional_losses_969478?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_966941input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_conv_block_5_layer_call_fn_967257
-__inference_conv_block_5_layer_call_fn_969870
-__inference_conv_block_5_layer_call_fn_969885
-__inference_conv_block_5_layer_call_fn_967317?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_conv_block_5_layer_call_and_return_conditional_losses_969907
H__inference_conv_block_5_layer_call_and_return_conditional_losses_969929
H__inference_conv_block_5_layer_call_and_return_conditional_losses_967332
H__inference_conv_block_5_layer_call_and_return_conditional_losses_967347?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_conv_block_6_layer_call_fn_967663
-__inference_conv_block_6_layer_call_fn_969944
-__inference_conv_block_6_layer_call_fn_969959
-__inference_conv_block_6_layer_call_fn_967723?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_conv_block_6_layer_call_and_return_conditional_losses_969981
H__inference_conv_block_6_layer_call_and_return_conditional_losses_970003
H__inference_conv_block_6_layer_call_and_return_conditional_losses_967738
H__inference_conv_block_6_layer_call_and_return_conditional_losses_967753?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_conv_block_7_layer_call_fn_968069
-__inference_conv_block_7_layer_call_fn_970018
-__inference_conv_block_7_layer_call_fn_970033
-__inference_conv_block_7_layer_call_fn_968129?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_conv_block_7_layer_call_and_return_conditional_losses_970055
H__inference_conv_block_7_layer_call_and_return_conditional_losses_970077
H__inference_conv_block_7_layer_call_and_return_conditional_losses_968144
H__inference_conv_block_7_layer_call_and_return_conditional_losses_968159?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_conv_block_8_layer_call_fn_968475
-__inference_conv_block_8_layer_call_fn_970092
-__inference_conv_block_8_layer_call_fn_970107
-__inference_conv_block_8_layer_call_fn_968535?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_conv_block_8_layer_call_and_return_conditional_losses_970129
H__inference_conv_block_8_layer_call_and_return_conditional_losses_970151
H__inference_conv_block_8_layer_call_and_return_conditional_losses_968550
H__inference_conv_block_8_layer_call_and_return_conditional_losses_968565?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_conv_block_9_layer_call_fn_968881
-__inference_conv_block_9_layer_call_fn_970166
-__inference_conv_block_9_layer_call_fn_970181
-__inference_conv_block_9_layer_call_fn_968941?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_conv_block_9_layer_call_and_return_conditional_losses_970203
H__inference_conv_block_9_layer_call_and_return_conditional_losses_970225
H__inference_conv_block_9_layer_call_and_return_conditional_losses_968956
H__inference_conv_block_9_layer_call_and_return_conditional_losses_968971?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_conv2d_10_layer_call_fn_970232?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_10_layer_call_and_return_conditional_losses_970240?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_969539input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_sequential_9_layer_call_fn_967071
-__inference_sequential_9_layer_call_fn_970255
-__inference_sequential_9_layer_call_fn_970270
-__inference_sequential_9_layer_call_fn_967191?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_sequential_9_layer_call_and_return_conditional_losses_970292
H__inference_sequential_9_layer_call_and_return_conditional_losses_970314
H__inference_sequential_9_layer_call_and_return_conditional_losses_967208
H__inference_sequential_9_layer_call_and_return_conditional_losses_967225?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_sequential_10_layer_call_fn_967477
.__inference_sequential_10_layer_call_fn_970329
.__inference_sequential_10_layer_call_fn_970344
.__inference_sequential_10_layer_call_fn_967597?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_sequential_10_layer_call_and_return_conditional_losses_970366
I__inference_sequential_10_layer_call_and_return_conditional_losses_970388
I__inference_sequential_10_layer_call_and_return_conditional_losses_967614
I__inference_sequential_10_layer_call_and_return_conditional_losses_967631?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_sequential_11_layer_call_fn_967883
.__inference_sequential_11_layer_call_fn_970403
.__inference_sequential_11_layer_call_fn_970418
.__inference_sequential_11_layer_call_fn_968003?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_sequential_11_layer_call_and_return_conditional_losses_970440
I__inference_sequential_11_layer_call_and_return_conditional_losses_970462
I__inference_sequential_11_layer_call_and_return_conditional_losses_968020
I__inference_sequential_11_layer_call_and_return_conditional_losses_968037?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_sequential_12_layer_call_fn_968289
.__inference_sequential_12_layer_call_fn_970477
.__inference_sequential_12_layer_call_fn_970492
.__inference_sequential_12_layer_call_fn_968409?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_sequential_12_layer_call_and_return_conditional_losses_970514
I__inference_sequential_12_layer_call_and_return_conditional_losses_970536
I__inference_sequential_12_layer_call_and_return_conditional_losses_968426
I__inference_sequential_12_layer_call_and_return_conditional_losses_968443?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_sequential_13_layer_call_fn_968695
.__inference_sequential_13_layer_call_fn_970551
.__inference_sequential_13_layer_call_fn_970566
.__inference_sequential_13_layer_call_fn_968815?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_sequential_13_layer_call_and_return_conditional_losses_970588
I__inference_sequential_13_layer_call_and_return_conditional_losses_970610
I__inference_sequential_13_layer_call_and_return_conditional_losses_968832
I__inference_sequential_13_layer_call_and_return_conditional_losses_968849?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_conv2d_5_layer_call_fn_970617?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_5_layer_call_and_return_conditional_losses_970624?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
6__inference_batch_normalization_9_layer_call_fn_970637
6__inference_batch_normalization_9_layer_call_fn_970650
6__inference_batch_normalization_9_layer_call_fn_970663
6__inference_batch_normalization_9_layer_call_fn_970676?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_970694
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_970712
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_970730
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_970748?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_leaky_re_lu_5_layer_call_fn_970753?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_970758?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_6_layer_call_fn_970765?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_6_layer_call_and_return_conditional_losses_970772?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_batch_normalization_10_layer_call_fn_970785
7__inference_batch_normalization_10_layer_call_fn_970798
7__inference_batch_normalization_10_layer_call_fn_970811
7__inference_batch_normalization_10_layer_call_fn_970824?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_970842
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_970860
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_970878
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_970896?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_leaky_re_lu_6_layer_call_fn_970901?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_970906?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_7_layer_call_fn_970913?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_7_layer_call_and_return_conditional_losses_970920?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_batch_normalization_11_layer_call_fn_970933
7__inference_batch_normalization_11_layer_call_fn_970946
7__inference_batch_normalization_11_layer_call_fn_970959
7__inference_batch_normalization_11_layer_call_fn_970972?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_970990
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_971008
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_971026
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_971044?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_leaky_re_lu_7_layer_call_fn_971049?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_971054?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_8_layer_call_fn_971061?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_8_layer_call_and_return_conditional_losses_971068?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_batch_normalization_12_layer_call_fn_971081
7__inference_batch_normalization_12_layer_call_fn_971094
7__inference_batch_normalization_12_layer_call_fn_971107
7__inference_batch_normalization_12_layer_call_fn_971120?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_971138
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_971156
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_971174
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_971192?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_leaky_re_lu_8_layer_call_fn_971197?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_971202?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_9_layer_call_fn_971209?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_9_layer_call_and_return_conditional_losses_971216?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_batch_normalization_13_layer_call_fn_971229
7__inference_batch_normalization_13_layer_call_fn_971242
7__inference_batch_normalization_13_layer_call_fn_971255
7__inference_batch_normalization_13_layer_call_fn_971268?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_971286
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_971304
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_971322
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_971340?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_leaky_re_lu_9_layer_call_fn_971345?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_971350?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_966941?*+,-./0123456789:;<=>?@AB%:?7
0?-
+?(
input_1???????????
? "s?p
6
output_1*?'
output_1?????????
6
output_2*?'
output_2?????????d?
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_970842?0123N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_970860?0123N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_970878t0123<?9
2?/
)?&
inputs?????????  ?
p 
? ".?+
$?!
0?????????  ?
? ?
R__inference_batch_normalization_10_layer_call_and_return_conditional_losses_970896t0123<?9
2?/
)?&
inputs?????????  ?
p
? ".?+
$?!
0?????????  ?
? ?
7__inference_batch_normalization_10_layer_call_fn_970785?0123N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
7__inference_batch_normalization_10_layer_call_fn_970798?0123N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
7__inference_batch_normalization_10_layer_call_fn_970811g0123<?9
2?/
)?&
inputs?????????  ?
p 
? "!??????????  ??
7__inference_batch_normalization_10_layer_call_fn_970824g0123<?9
2?/
)?&
inputs?????????  ?
p
? "!??????????  ??
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_970990?5678N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_971008?5678N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_971026t5678<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
R__inference_batch_normalization_11_layer_call_and_return_conditional_losses_971044t5678<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
7__inference_batch_normalization_11_layer_call_fn_970933?5678N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
7__inference_batch_normalization_11_layer_call_fn_970946?5678N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
7__inference_batch_normalization_11_layer_call_fn_970959g5678<?9
2?/
)?&
inputs??????????
p 
? "!????????????
7__inference_batch_normalization_11_layer_call_fn_970972g5678<?9
2?/
)?&
inputs??????????
p
? "!????????????
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_971138?:;<=N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_971156?:;<=N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_971174t:;<=<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
R__inference_batch_normalization_12_layer_call_and_return_conditional_losses_971192t:;<=<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
7__inference_batch_normalization_12_layer_call_fn_971081?:;<=N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
7__inference_batch_normalization_12_layer_call_fn_971094?:;<=N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
7__inference_batch_normalization_12_layer_call_fn_971107g:;<=<?9
2?/
)?&
inputs??????????
p 
? "!????????????
7__inference_batch_normalization_12_layer_call_fn_971120g:;<=<?9
2?/
)?&
inputs??????????
p
? "!????????????
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_971286??@ABM?J
C?@
:?7
inputs+???????????????????????????d
p 
? "??<
5?2
0+???????????????????????????d
? ?
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_971304??@ABM?J
C?@
:?7
inputs+???????????????????????????d
p
? "??<
5?2
0+???????????????????????????d
? ?
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_971322r?@AB;?8
1?.
(?%
inputs?????????d
p 
? "-?*
#? 
0?????????d
? ?
R__inference_batch_normalization_13_layer_call_and_return_conditional_losses_971340r?@AB;?8
1?.
(?%
inputs?????????d
p
? "-?*
#? 
0?????????d
? ?
7__inference_batch_normalization_13_layer_call_fn_971229??@ABM?J
C?@
:?7
inputs+???????????????????????????d
p 
? "2?/+???????????????????????????d?
7__inference_batch_normalization_13_layer_call_fn_971242??@ABM?J
C?@
:?7
inputs+???????????????????????????d
p
? "2?/+???????????????????????????d?
7__inference_batch_normalization_13_layer_call_fn_971255e?@AB;?8
1?.
(?%
inputs?????????d
p 
? " ??????????d?
7__inference_batch_normalization_13_layer_call_fn_971268e?@AB;?8
1?.
(?%
inputs?????????d
p
? " ??????????d?
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_970694?+,-.M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_970712?+,-.M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_970730r+,-.;?8
1?.
(?%
inputs?????????@@@
p 
? "-?*
#? 
0?????????@@@
? ?
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_970748r+,-.;?8
1?.
(?%
inputs?????????@@@
p
? "-?*
#? 
0?????????@@@
? ?
6__inference_batch_normalization_9_layer_call_fn_970637?+,-.M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
6__inference_batch_normalization_9_layer_call_fn_970650?+,-.M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
6__inference_batch_normalization_9_layer_call_fn_970663e+,-.;?8
1?.
(?%
inputs?????????@@@
p 
? " ??????????@@@?
6__inference_batch_normalization_9_layer_call_fn_970676e+,-.;?8
1?.
(?%
inputs?????????@@@
p
? " ??????????@@@?
E__inference_conv2d_10_layer_call_and_return_conditional_losses_970240k%7?4
-?*
(?%
inputs?????????d
? "-?*
#? 
0?????????
? ?
*__inference_conv2d_10_layer_call_fn_970232^%7?4
-?*
(?%
inputs?????????d
? " ???????????
D__inference_conv2d_5_layer_call_and_return_conditional_losses_970624m*9?6
/?,
*?'
inputs???????????
? "-?*
#? 
0?????????@@@
? ?
)__inference_conv2d_5_layer_call_fn_970617`*9?6
/?,
*?'
inputs???????????
? " ??????????@@@?
D__inference_conv2d_6_layer_call_and_return_conditional_losses_970772l/7?4
-?*
(?%
inputs?????????@@@
? ".?+
$?!
0?????????  ?
? ?
)__inference_conv2d_6_layer_call_fn_970765_/7?4
-?*
(?%
inputs?????????@@@
? "!??????????  ??
D__inference_conv2d_7_layer_call_and_return_conditional_losses_970920m48?5
.?+
)?&
inputs?????????  ?
? ".?+
$?!
0??????????
? ?
)__inference_conv2d_7_layer_call_fn_970913`48?5
.?+
)?&
inputs?????????  ?
? "!????????????
D__inference_conv2d_8_layer_call_and_return_conditional_losses_971068m98?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
)__inference_conv2d_8_layer_call_fn_971061`98?5
.?+
)?&
inputs??????????
? "!????????????
D__inference_conv2d_9_layer_call_and_return_conditional_losses_971216l>8?5
.?+
)?&
inputs??????????
? "-?*
#? 
0?????????d
? ?
)__inference_conv2d_9_layer_call_fn_971209_>8?5
.?+
)?&
inputs??????????
? " ??????????d?
H__inference_conv_block_5_layer_call_and_return_conditional_losses_967332v*+,-.>?;
4?1
+?(
input_1???????????
p 
? "-?*
#? 
0?????????@@@
? ?
H__inference_conv_block_5_layer_call_and_return_conditional_losses_967347v*+,-.>?;
4?1
+?(
input_1???????????
p
? "-?*
#? 
0?????????@@@
? ?
H__inference_conv_block_5_layer_call_and_return_conditional_losses_969907u*+,-.=?:
3?0
*?'
inputs???????????
p 
? "-?*
#? 
0?????????@@@
? ?
H__inference_conv_block_5_layer_call_and_return_conditional_losses_969929u*+,-.=?:
3?0
*?'
inputs???????????
p
? "-?*
#? 
0?????????@@@
? ?
-__inference_conv_block_5_layer_call_fn_967257i*+,-.>?;
4?1
+?(
input_1???????????
p 
? " ??????????@@@?
-__inference_conv_block_5_layer_call_fn_967317i*+,-.>?;
4?1
+?(
input_1???????????
p
? " ??????????@@@?
-__inference_conv_block_5_layer_call_fn_969870h*+,-.=?:
3?0
*?'
inputs???????????
p 
? " ??????????@@@?
-__inference_conv_block_5_layer_call_fn_969885h*+,-.=?:
3?0
*?'
inputs???????????
p
? " ??????????@@@?
H__inference_conv_block_6_layer_call_and_return_conditional_losses_967738u/0123<?9
2?/
)?&
input_1?????????@@@
p 
? ".?+
$?!
0?????????  ?
? ?
H__inference_conv_block_6_layer_call_and_return_conditional_losses_967753u/0123<?9
2?/
)?&
input_1?????????@@@
p
? ".?+
$?!
0?????????  ?
? ?
H__inference_conv_block_6_layer_call_and_return_conditional_losses_969981t/0123;?8
1?.
(?%
inputs?????????@@@
p 
? ".?+
$?!
0?????????  ?
? ?
H__inference_conv_block_6_layer_call_and_return_conditional_losses_970003t/0123;?8
1?.
(?%
inputs?????????@@@
p
? ".?+
$?!
0?????????  ?
? ?
-__inference_conv_block_6_layer_call_fn_967663h/0123<?9
2?/
)?&
input_1?????????@@@
p 
? "!??????????  ??
-__inference_conv_block_6_layer_call_fn_967723h/0123<?9
2?/
)?&
input_1?????????@@@
p
? "!??????????  ??
-__inference_conv_block_6_layer_call_fn_969944g/0123;?8
1?.
(?%
inputs?????????@@@
p 
? "!??????????  ??
-__inference_conv_block_6_layer_call_fn_969959g/0123;?8
1?.
(?%
inputs?????????@@@
p
? "!??????????  ??
H__inference_conv_block_7_layer_call_and_return_conditional_losses_968144v45678=?:
3?0
*?'
input_1?????????  ?
p 
? ".?+
$?!
0??????????
? ?
H__inference_conv_block_7_layer_call_and_return_conditional_losses_968159v45678=?:
3?0
*?'
input_1?????????  ?
p
? ".?+
$?!
0??????????
? ?
H__inference_conv_block_7_layer_call_and_return_conditional_losses_970055u45678<?9
2?/
)?&
inputs?????????  ?
p 
? ".?+
$?!
0??????????
? ?
H__inference_conv_block_7_layer_call_and_return_conditional_losses_970077u45678<?9
2?/
)?&
inputs?????????  ?
p
? ".?+
$?!
0??????????
? ?
-__inference_conv_block_7_layer_call_fn_968069i45678=?:
3?0
*?'
input_1?????????  ?
p 
? "!????????????
-__inference_conv_block_7_layer_call_fn_968129i45678=?:
3?0
*?'
input_1?????????  ?
p
? "!????????????
-__inference_conv_block_7_layer_call_fn_970018h45678<?9
2?/
)?&
inputs?????????  ?
p 
? "!????????????
-__inference_conv_block_7_layer_call_fn_970033h45678<?9
2?/
)?&
inputs?????????  ?
p
? "!????????????
H__inference_conv_block_8_layer_call_and_return_conditional_losses_968550v9:;<==?:
3?0
*?'
input_1??????????
p 
? ".?+
$?!
0??????????
? ?
H__inference_conv_block_8_layer_call_and_return_conditional_losses_968565v9:;<==?:
3?0
*?'
input_1??????????
p
? ".?+
$?!
0??????????
? ?
H__inference_conv_block_8_layer_call_and_return_conditional_losses_970129u9:;<=<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
H__inference_conv_block_8_layer_call_and_return_conditional_losses_970151u9:;<=<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
-__inference_conv_block_8_layer_call_fn_968475i9:;<==?:
3?0
*?'
input_1??????????
p 
? "!????????????
-__inference_conv_block_8_layer_call_fn_968535i9:;<==?:
3?0
*?'
input_1??????????
p
? "!????????????
-__inference_conv_block_8_layer_call_fn_970092h9:;<=<?9
2?/
)?&
inputs??????????
p 
? "!????????????
-__inference_conv_block_8_layer_call_fn_970107h9:;<=<?9
2?/
)?&
inputs??????????
p
? "!????????????
H__inference_conv_block_9_layer_call_and_return_conditional_losses_968956u>?@AB=?:
3?0
*?'
input_1??????????
p 
? "-?*
#? 
0?????????d
? ?
H__inference_conv_block_9_layer_call_and_return_conditional_losses_968971u>?@AB=?:
3?0
*?'
input_1??????????
p
? "-?*
#? 
0?????????d
? ?
H__inference_conv_block_9_layer_call_and_return_conditional_losses_970203t>?@AB<?9
2?/
)?&
inputs??????????
p 
? "-?*
#? 
0?????????d
? ?
H__inference_conv_block_9_layer_call_and_return_conditional_losses_970225t>?@AB<?9
2?/
)?&
inputs??????????
p
? "-?*
#? 
0?????????d
? ?
-__inference_conv_block_9_layer_call_fn_968881h>?@AB=?:
3?0
*?'
input_1??????????
p 
? " ??????????d?
-__inference_conv_block_9_layer_call_fn_968941h>?@AB=?:
3?0
*?'
input_1??????????
p
? " ??????????d?
-__inference_conv_block_9_layer_call_fn_970166g>?@AB<?9
2?/
)?&
inputs??????????
p 
? " ??????????d?
-__inference_conv_block_9_layer_call_fn_970181g>?@AB<?9
2?/
)?&
inputs??????????
p
? " ??????????d?
I__inference_discriminator_layer_call_and_return_conditional_losses_969415?*+,-./0123456789:;<=>?@AB%>?;
4?1
+?(
input_1???????????
p 
? "[?X
Q?N
%?"
0/0?????????
%?"
0/1?????????d
? ?
I__inference_discriminator_layer_call_and_return_conditional_losses_969478?*+,-./0123456789:;<=>?@AB%>?;
4?1
+?(
input_1???????????
p
? "[?X
Q?N
%?"
0/0?????????
%?"
0/1?????????d
? ?
I__inference_discriminator_layer_call_and_return_conditional_losses_969756?*+,-./0123456789:;<=>?@AB%=?:
3?0
*?'
inputs???????????
p 
? "[?X
Q?N
%?"
0/0?????????
%?"
0/1?????????d
? ?
I__inference_discriminator_layer_call_and_return_conditional_losses_969855?*+,-./0123456789:;<=>?@AB%=?:
3?0
*?'
inputs???????????
p
? "[?X
Q?N
%?"
0/0?????????
%?"
0/1?????????d
? ?
.__inference_discriminator_layer_call_fn_969104?*+,-./0123456789:;<=>?@AB%>?;
4?1
+?(
input_1???????????
p 
? "M?J
#? 
0?????????
#? 
1?????????d?
.__inference_discriminator_layer_call_fn_969352?*+,-./0123456789:;<=>?@AB%>?;
4?1
+?(
input_1???????????
p
? "M?J
#? 
0?????????
#? 
1?????????d?
.__inference_discriminator_layer_call_fn_969598?*+,-./0123456789:;<=>?@AB%=?:
3?0
*?'
inputs???????????
p 
? "M?J
#? 
0?????????
#? 
1?????????d?
.__inference_discriminator_layer_call_fn_969657?*+,-./0123456789:;<=>?@AB%=?:
3?0
*?'
inputs???????????
p
? "M?J
#? 
0?????????
#? 
1?????????d?
I__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_970758h7?4
-?*
(?%
inputs?????????@@@
? "-?*
#? 
0?????????@@@
? ?
.__inference_leaky_re_lu_5_layer_call_fn_970753[7?4
-?*
(?%
inputs?????????@@@
? " ??????????@@@?
I__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_970906j8?5
.?+
)?&
inputs?????????  ?
? ".?+
$?!
0?????????  ?
? ?
.__inference_leaky_re_lu_6_layer_call_fn_970901]8?5
.?+
)?&
inputs?????????  ?
? "!??????????  ??
I__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_971054j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
.__inference_leaky_re_lu_7_layer_call_fn_971049]8?5
.?+
)?&
inputs??????????
? "!????????????
I__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_971202j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
.__inference_leaky_re_lu_8_layer_call_fn_971197]8?5
.?+
)?&
inputs??????????
? "!????????????
I__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_971350h7?4
-?*
(?%
inputs?????????d
? "-?*
#? 
0?????????d
? ?
.__inference_leaky_re_lu_9_layer_call_fn_971345[7?4
-?*
(?%
inputs?????????d
? " ??????????d?
I__inference_sequential_10_layer_call_and_return_conditional_losses_967614?/0123G?D
=?:
0?-
conv2d_6_input?????????@@@
p 

 
? ".?+
$?!
0?????????  ?
? ?
I__inference_sequential_10_layer_call_and_return_conditional_losses_967631?/0123G?D
=?:
0?-
conv2d_6_input?????????@@@
p

 
? ".?+
$?!
0?????????  ?
? ?
I__inference_sequential_10_layer_call_and_return_conditional_losses_970366x/0123??<
5?2
(?%
inputs?????????@@@
p 

 
? ".?+
$?!
0?????????  ?
? ?
I__inference_sequential_10_layer_call_and_return_conditional_losses_970388x/0123??<
5?2
(?%
inputs?????????@@@
p

 
? ".?+
$?!
0?????????  ?
? ?
.__inference_sequential_10_layer_call_fn_967477s/0123G?D
=?:
0?-
conv2d_6_input?????????@@@
p 

 
? "!??????????  ??
.__inference_sequential_10_layer_call_fn_967597s/0123G?D
=?:
0?-
conv2d_6_input?????????@@@
p

 
? "!??????????  ??
.__inference_sequential_10_layer_call_fn_970329k/0123??<
5?2
(?%
inputs?????????@@@
p 

 
? "!??????????  ??
.__inference_sequential_10_layer_call_fn_970344k/0123??<
5?2
(?%
inputs?????????@@@
p

 
? "!??????????  ??
I__inference_sequential_11_layer_call_and_return_conditional_losses_968020?45678H?E
>?;
1?.
conv2d_7_input?????????  ?
p 

 
? ".?+
$?!
0??????????
? ?
I__inference_sequential_11_layer_call_and_return_conditional_losses_968037?45678H?E
>?;
1?.
conv2d_7_input?????????  ?
p

 
? ".?+
$?!
0??????????
? ?
I__inference_sequential_11_layer_call_and_return_conditional_losses_970440y45678@?=
6?3
)?&
inputs?????????  ?
p 

 
? ".?+
$?!
0??????????
? ?
I__inference_sequential_11_layer_call_and_return_conditional_losses_970462y45678@?=
6?3
)?&
inputs?????????  ?
p

 
? ".?+
$?!
0??????????
? ?
.__inference_sequential_11_layer_call_fn_967883t45678H?E
>?;
1?.
conv2d_7_input?????????  ?
p 

 
? "!????????????
.__inference_sequential_11_layer_call_fn_968003t45678H?E
>?;
1?.
conv2d_7_input?????????  ?
p

 
? "!????????????
.__inference_sequential_11_layer_call_fn_970403l45678@?=
6?3
)?&
inputs?????????  ?
p 

 
? "!????????????
.__inference_sequential_11_layer_call_fn_970418l45678@?=
6?3
)?&
inputs?????????  ?
p

 
? "!????????????
I__inference_sequential_12_layer_call_and_return_conditional_losses_968426?9:;<=H?E
>?;
1?.
conv2d_8_input??????????
p 

 
? ".?+
$?!
0??????????
? ?
I__inference_sequential_12_layer_call_and_return_conditional_losses_968443?9:;<=H?E
>?;
1?.
conv2d_8_input??????????
p

 
? ".?+
$?!
0??????????
? ?
I__inference_sequential_12_layer_call_and_return_conditional_losses_970514y9:;<=@?=
6?3
)?&
inputs??????????
p 

 
? ".?+
$?!
0??????????
? ?
I__inference_sequential_12_layer_call_and_return_conditional_losses_970536y9:;<=@?=
6?3
)?&
inputs??????????
p

 
? ".?+
$?!
0??????????
? ?
.__inference_sequential_12_layer_call_fn_968289t9:;<=H?E
>?;
1?.
conv2d_8_input??????????
p 

 
? "!????????????
.__inference_sequential_12_layer_call_fn_968409t9:;<=H?E
>?;
1?.
conv2d_8_input??????????
p

 
? "!????????????
.__inference_sequential_12_layer_call_fn_970477l9:;<=@?=
6?3
)?&
inputs??????????
p 

 
? "!????????????
.__inference_sequential_12_layer_call_fn_970492l9:;<=@?=
6?3
)?&
inputs??????????
p

 
? "!????????????
I__inference_sequential_13_layer_call_and_return_conditional_losses_968832?>?@ABH?E
>?;
1?.
conv2d_9_input??????????
p 

 
? "-?*
#? 
0?????????d
? ?
I__inference_sequential_13_layer_call_and_return_conditional_losses_968849?>?@ABH?E
>?;
1?.
conv2d_9_input??????????
p

 
? "-?*
#? 
0?????????d
? ?
I__inference_sequential_13_layer_call_and_return_conditional_losses_970588x>?@AB@?=
6?3
)?&
inputs??????????
p 

 
? "-?*
#? 
0?????????d
? ?
I__inference_sequential_13_layer_call_and_return_conditional_losses_970610x>?@AB@?=
6?3
)?&
inputs??????????
p

 
? "-?*
#? 
0?????????d
? ?
.__inference_sequential_13_layer_call_fn_968695s>?@ABH?E
>?;
1?.
conv2d_9_input??????????
p 

 
? " ??????????d?
.__inference_sequential_13_layer_call_fn_968815s>?@ABH?E
>?;
1?.
conv2d_9_input??????????
p

 
? " ??????????d?
.__inference_sequential_13_layer_call_fn_970551k>?@AB@?=
6?3
)?&
inputs??????????
p 

 
? " ??????????d?
.__inference_sequential_13_layer_call_fn_970566k>?@AB@?=
6?3
)?&
inputs??????????
p

 
? " ??????????d?
H__inference_sequential_9_layer_call_and_return_conditional_losses_967208?*+,-.I?F
??<
2?/
conv2d_5_input???????????
p 

 
? "-?*
#? 
0?????????@@@
? ?
H__inference_sequential_9_layer_call_and_return_conditional_losses_967225?*+,-.I?F
??<
2?/
conv2d_5_input???????????
p

 
? "-?*
#? 
0?????????@@@
? ?
H__inference_sequential_9_layer_call_and_return_conditional_losses_970292y*+,-.A?>
7?4
*?'
inputs???????????
p 

 
? "-?*
#? 
0?????????@@@
? ?
H__inference_sequential_9_layer_call_and_return_conditional_losses_970314y*+,-.A?>
7?4
*?'
inputs???????????
p

 
? "-?*
#? 
0?????????@@@
? ?
-__inference_sequential_9_layer_call_fn_967071t*+,-.I?F
??<
2?/
conv2d_5_input???????????
p 

 
? " ??????????@@@?
-__inference_sequential_9_layer_call_fn_967191t*+,-.I?F
??<
2?/
conv2d_5_input???????????
p

 
? " ??????????@@@?
-__inference_sequential_9_layer_call_fn_970255l*+,-.A?>
7?4
*?'
inputs???????????
p 

 
? " ??????????@@@?
-__inference_sequential_9_layer_call_fn_970270l*+,-.A?>
7?4
*?'
inputs???????????
p

 
? " ??????????@@@?
$__inference_signature_wrapper_969539?*+,-./0123456789:;<=>?@AB%E?B
? 
;?8
6
input_1+?(
input_1???????????"s?p
6
output_1*?'
output_1?????????
6
output_2*?'
output_2?????????d