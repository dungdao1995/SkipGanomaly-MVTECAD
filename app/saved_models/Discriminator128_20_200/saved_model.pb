??
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
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
 ?"serve*2.8.02v2.8.0-0-g3f878cff5b68??
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
ˏ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
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
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
?

conv_layer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?

conv_layer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?

conv_layer
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses*
?
$
conv_layer
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses*
?
+
conv_layer
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses*
?

2kernel
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses*
?
90
:1
;2
<3
=4
>5
?6
@7
A8
B9
C10
D11
E12
F13
G14
H15
I16
J17
K18
L19
M20
N21
O22
P23
Q24
225*
z
90
:1
;2
>3
?4
@5
C6
D7
E8
H9
I10
J11
M12
N13
O14
215*
* 
?
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Wserving_default* 
?
Xlayer_with_weights-0
Xlayer-0
Ylayer_with_weights-1
Ylayer-1
Zlayer-2
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses*
'
90
:1
;2
<3
=4*

90
:1
;2*
* 
?
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
?
flayer_with_weights-0
flayer-0
glayer_with_weights-1
glayer-1
hlayer-2
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses*
'
>0
?1
@2
A3
B4*

>0
?1
@2*
* 
?
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
?
tlayer_with_weights-0
tlayer-0
ulayer_with_weights-1
ulayer-1
vlayer-2
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses*
'
C0
D1
E2
F3
G4*

C0
D1
E2*
* 
?
}non_trainable_variables

~layers
metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*
* 
* 
?
?layer_with_weights-0
?layer-0
?layer_with_weights-1
?layer-1
?layer-2
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
'
H0
I1
J2
K3
L4*

H0
I1
J2*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*
* 
* 
?
?layer_with_weights-0
?layer-0
?layer_with_weights-1
?layer-1
?layer-2
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
'
M0
N1
O2
P3
Q4*

M0
N1
O2*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*
* 
* 
a[
VARIABLE_VALUEdiscriminator/conv2d_10/kernel)outputs/kernel/.ATTRIBUTES/VARIABLE_VALUE*

20*

20*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses*
* 
* 
OI
VARIABLE_VALUEconv2d_5/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbatch_normalization_9/gamma&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEbatch_normalization_9/beta&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!batch_normalization_9/moving_mean&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%batch_normalization_9/moving_variance&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_6/kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEbatch_normalization_10/gamma&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbatch_normalization_10/beta&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"batch_normalization_10/moving_mean&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&batch_normalization_10/moving_variance&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_7/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEbatch_normalization_11/gamma'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEbatch_normalization_11/beta'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"batch_normalization_11/moving_mean'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&batch_normalization_11/moving_variance'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_8/kernel'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEbatch_normalization_12/gamma'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEbatch_normalization_12/beta'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"batch_normalization_12/moving_mean'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&batch_normalization_12/moving_variance'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_9/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEbatch_normalization_13/gamma'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEbatch_normalization_13/beta'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"batch_normalization_13/moving_mean'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&batch_normalization_13/moving_variance'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
J
<0
=1
A2
B3
F4
G5
K6
L7
P8
Q9*
.
0
1
2
3
4
5*
* 
* 
* 
* 
?

9kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
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
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
'
90
:1
;2
<3
=4*

90
:1
;2*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses*
* 
* 

<0
=1*

0*
* 
* 
* 
?

>kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
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
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
'
>0
?1
@2
A3
B4*

>0
?1
@2*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses*
* 
* 

A0
B1*

0*
* 
* 
* 
?

Ckernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
	?axis
	Dgamma
Ebeta
Fmoving_mean
Gmoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
'
C0
D1
E2
F3
G4*

C0
D1
E2*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses*
* 
* 

F0
G1*

0*
* 
* 
* 
?

Hkernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
	?axis
	Igamma
Jbeta
Kmoving_mean
Lmoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
'
H0
I1
J2
K3
L4*

H0
I1
J2*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

K0
L1*

$0*
* 
* 
* 
?

Mkernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
	?axis
	Ngamma
Obeta
Pmoving_mean
Qmoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
'
M0
N1
O2
P3
Q4*

M0
N1
O2*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 

P0
Q1*

+0*
* 
* 
* 
* 
* 
* 
* 
* 

90*

90*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
 
:0
;1
<2
=3*

:0
;1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 

<0
=1*

X0
Y1
Z2*
* 
* 
* 

>0*

>0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
 
?0
@1
A2
B3*

?0
@1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 

A0
B1*

f0
g1
h2*
* 
* 
* 

C0*

C0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
 
D0
E1
F2
G3*

D0
E1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 

F0
G1*

t0
u1
v2*
* 
* 
* 

H0*

H0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
 
I0
J1
K2
L3*

I0
J1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 

K0
L1*

?0
?1
?2*
* 
* 
* 

M0*

M0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
 
N0
O1
P2
Q3*

N0
O1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 

P0
Q1*

?0
?1
?2*
* 
* 
* 
* 
* 
* 
* 
* 

<0
=1*
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
A0
B1*
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
F0
G1*
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
K0
L1*
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
P0
Q1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
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
*0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference_signature_wrapper_3835436
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
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_save_3836723
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
 *0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__traced_restore_3836811??
??
?(
"__inference__wrapped_model_3832832
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
?
?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3836249

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
?
?
J__inference_sequential_11_layer_call_and_return_conditional_losses_3833725
conv2d_7_input,
conv2d_7_3833711:??-
batch_normalization_11_3833714:	?-
batch_normalization_11_3833716:	?-
batch_normalization_11_3833718:	?-
batch_normalization_11_3833720:	?
identity??.batch_normalization_11/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCallconv2d_7_inputconv2d_7_3833711*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_3833598?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_11_3833714batch_normalization_11_3833716batch_normalization_11_3833718batch_normalization_11_3833720*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_3833542?
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
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_3833616~
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
.__inference_sequential_9_layer_call_fn_3835851

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
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_9_layer_call_and_return_conditional_losses_3832992w
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
?
8__inference_batch_normalization_13_layer_call_fn_3836575

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
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_3834261?
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
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_3836525

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
?
?
.__inference_conv_block_6_layer_call_fn_3833430
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
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_6_layer_call_and_return_conditional_losses_3833417x
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
?

?
I__inference_conv_block_6_layer_call_and_return_conditional_losses_3833417

inputs0
sequential_10_3833405:@?$
sequential_10_3833407:	?$
sequential_10_3833409:	?$
sequential_10_3833411:	?$
sequential_10_3833413:	?
identity??%sequential_10/StatefulPartitionedCall?
%sequential_10/StatefulPartitionedCallStatefulPartitionedCallinputssequential_10_3833405sequential_10_3833407sequential_10_3833409sequential_10_3833411sequential_10_3833413*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_10_layer_call_and_return_conditional_losses_3833275?
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
?
?
/__inference_discriminator_layer_call_fn_3834933
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
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_discriminator_layer_call_and_return_conditional_losses_3834817w
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
?
?
/__inference_sequential_10_layer_call_fn_3835925

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
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_10_layer_call_and_return_conditional_losses_3833336x
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
?
K
/__inference_leaky_re_lu_8_layer_call_fn_3836530

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
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_3833960i
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
?i
?
#__inference__traced_restore_3836811
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
?
?
.__inference_conv_block_5_layer_call_fn_3833146
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
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_5_layer_call_and_return_conditional_losses_3833118w
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
?
?
*__inference_conv2d_5_layer_call_fn_3836198

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
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_3832910w
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
?
?
/__inference_sequential_11_layer_call_fn_3835984

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
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_11_layer_call_and_return_conditional_losses_3833619x
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
?
?
.__inference_conv_block_8_layer_call_fn_3835673

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
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_8_layer_call_and_return_conditional_losses_3834105x
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
?
?
I__inference_conv_block_7_layer_call_and_return_conditional_losses_3835636

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
?
?
I__inference_sequential_9_layer_call_and_return_conditional_losses_3833037
conv2d_5_input*
conv2d_5_3833023:@+
batch_normalization_9_3833026:@+
batch_normalization_9_3833028:@+
batch_normalization_9_3833030:@+
batch_normalization_9_3833032:@
identity??-batch_normalization_9/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputconv2d_5_3833023*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_3832910?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_9_3833026batch_normalization_9_3833028batch_normalization_9_3833030batch_normalization_9_3833032*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3832854?
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
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_3832928}
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
?
?
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_3834230

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
?
?
.__inference_conv_block_7_layer_call_fn_3833774
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
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_7_layer_call_and_return_conditional_losses_3833761x
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
?
?
J__inference_sequential_13_layer_call_and_return_conditional_losses_3834430
conv2d_9_input+
conv2d_9_3834416:?d,
batch_normalization_13_3834419:d,
batch_normalization_13_3834421:d,
batch_normalization_13_3834423:d,
batch_normalization_13_3834425:d
identity??.batch_normalization_13/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallconv2d_9_inputconv2d_9_3834416*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_9_layer_call_and_return_conditional_losses_3834286?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0batch_normalization_13_3834419batch_normalization_13_3834421batch_normalization_13_3834423batch_normalization_13_3834425*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_3834261?
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
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_3834304}
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
?
?
E__inference_conv2d_6_layer_call_and_return_conditional_losses_3833254

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
?
?
I__inference_conv_block_6_layer_call_and_return_conditional_losses_3835562

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
˫
?#
J__inference_discriminator_layer_call_and_return_conditional_losses_3835276

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
?	
?
/__inference_sequential_11_layer_call_fn_3833708
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
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_11_layer_call_and_return_conditional_losses_3833680x
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
?
f
J__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_3836277

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
?	
?
/__inference_sequential_12_layer_call_fn_3833976
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
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_12_layer_call_and_return_conditional_losses_3833963x
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
/__inference_sequential_13_layer_call_fn_3834320
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
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_3834307w
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
?
f
J__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_3834304

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
?
8__inference_batch_normalization_10_layer_call_fn_3836304

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
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_3833198?
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
?
?
/__inference_discriminator_layer_call_fn_3835177

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
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_discriminator_layer_call_and_return_conditional_losses_3834817w
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
?
?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3832885

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
?

?
I__inference_conv_block_8_layer_call_and_return_conditional_losses_3834193
input_11
sequential_12_3834181:??$
sequential_12_3834183:	?$
sequential_12_3834185:	?$
sequential_12_3834187:	?$
sequential_12_3834189:	?
identity??%sequential_12/StatefulPartitionedCall?
%sequential_12/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_12_3834181sequential_12_3834183sequential_12_3834185sequential_12_3834187sequential_12_3834189*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_12_layer_call_and_return_conditional_losses_3833963?
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
?!
?
J__inference_sequential_13_layer_call_and_return_conditional_losses_3836191

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
?
?
/__inference_sequential_11_layer_call_fn_3835999

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
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_11_layer_call_and_return_conditional_losses_3833680x
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
I__inference_conv_block_8_layer_call_and_return_conditional_losses_3834105

inputs1
sequential_12_3834093:??$
sequential_12_3834095:	?$
sequential_12_3834097:	?$
sequential_12_3834099:	?$
sequential_12_3834101:	?
identity??%sequential_12/StatefulPartitionedCall?
%sequential_12/StatefulPartitionedCallStatefulPartitionedCallinputssequential_12_3834093sequential_12_3834095sequential_12_3834097sequential_12_3834099sequential_12_3834101*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_12_layer_call_and_return_conditional_losses_3833963?
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
?
?
E__inference_conv2d_7_layer_call_and_return_conditional_losses_3836377

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
?
?
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_3836611

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
?
f
J__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_3833272

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
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_3834261

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
8__inference_batch_normalization_12_layer_call_fn_3836489

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
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_3833917?
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
?
?
I__inference_sequential_9_layer_call_and_return_conditional_losses_3832992

inputs*
conv2d_5_3832978:@+
batch_normalization_9_3832981:@+
batch_normalization_9_3832983:@+
batch_normalization_9_3832985:@+
batch_normalization_9_3832987:@
identity??-batch_normalization_9/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_3832978*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_3832910?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_9_3832981batch_normalization_9_3832983batch_normalization_9_3832985batch_normalization_9_3832987*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3832885?
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
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_3832928}
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
?
?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_3836421

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
?
?
.__inference_conv_block_8_layer_call_fn_3834178
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
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_8_layer_call_and_return_conditional_losses_3834150x
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
?
K
/__inference_leaky_re_lu_6_layer_call_fn_3836358

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
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_3833272i
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
?
?
E__inference_conv2d_8_layer_call_and_return_conditional_losses_3836463

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
?
?
J__inference_sequential_13_layer_call_and_return_conditional_losses_3834307

inputs+
conv2d_9_3834287:?d,
batch_normalization_13_3834290:d,
batch_normalization_13_3834292:d,
batch_normalization_13_3834294:d,
batch_normalization_13_3834296:d
identity??.batch_normalization_13/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_9_3834287*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_9_layer_call_and_return_conditional_losses_3834286?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0batch_normalization_13_3834290batch_normalization_13_3834292batch_normalization_13_3834294batch_normalization_13_3834296*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_3834230?
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
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_3834304}
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
?
?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_3833886

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
?

?
I__inference_conv_block_5_layer_call_and_return_conditional_losses_3833118

inputs.
sequential_9_3833106:@"
sequential_9_3833108:@"
sequential_9_3833110:@"
sequential_9_3833112:@"
sequential_9_3833114:@
identity??$sequential_9/StatefulPartitionedCall?
$sequential_9/StatefulPartitionedCallStatefulPartitionedCallinputssequential_9_3833106sequential_9_3833108sequential_9_3833110sequential_9_3833112sequential_9_3833114*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_9_layer_call_and_return_conditional_losses_3832992?
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
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_3833917

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
?
?
/__inference_sequential_13_layer_call_fn_3836132

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
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_3834307w
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
?,
?

J__inference_discriminator_layer_call_and_return_conditional_losses_3834996
input_1.
conv_block_5_3834936:@"
conv_block_5_3834938:@"
conv_block_5_3834940:@"
conv_block_5_3834942:@"
conv_block_5_3834944:@/
conv_block_6_3834947:@?#
conv_block_6_3834949:	?#
conv_block_6_3834951:	?#
conv_block_6_3834953:	?#
conv_block_6_3834955:	?0
conv_block_7_3834958:??#
conv_block_7_3834960:	?#
conv_block_7_3834962:	?#
conv_block_7_3834964:	?#
conv_block_7_3834966:	?0
conv_block_8_3834969:??#
conv_block_8_3834971:	?#
conv_block_8_3834973:	?#
conv_block_8_3834975:	?#
conv_block_8_3834977:	?/
conv_block_9_3834980:?d"
conv_block_9_3834982:d"
conv_block_9_3834984:d"
conv_block_9_3834986:d"
conv_block_9_3834988:d+
conv2d_10_3834991:d
identity

identity_1??!conv2d_10/StatefulPartitionedCall?$conv_block_5/StatefulPartitionedCall?$conv_block_6/StatefulPartitionedCall?$conv_block_7/StatefulPartitionedCall?$conv_block_8/StatefulPartitionedCall?$conv_block_9/StatefulPartitionedCall?
$conv_block_5/StatefulPartitionedCallStatefulPartitionedCallinput_1conv_block_5_3834936conv_block_5_3834938conv_block_5_3834940conv_block_5_3834942conv_block_5_3834944*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_5_layer_call_and_return_conditional_losses_3833073?
$conv_block_6/StatefulPartitionedCallStatefulPartitionedCall-conv_block_5/StatefulPartitionedCall:output:0conv_block_6_3834947conv_block_6_3834949conv_block_6_3834951conv_block_6_3834953conv_block_6_3834955*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_6_layer_call_and_return_conditional_losses_3833417?
$conv_block_7/StatefulPartitionedCallStatefulPartitionedCall-conv_block_6/StatefulPartitionedCall:output:0conv_block_7_3834958conv_block_7_3834960conv_block_7_3834962conv_block_7_3834964conv_block_7_3834966*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_7_layer_call_and_return_conditional_losses_3833761?
$conv_block_8/StatefulPartitionedCallStatefulPartitionedCall-conv_block_7/StatefulPartitionedCall:output:0conv_block_8_3834969conv_block_8_3834971conv_block_8_3834973conv_block_8_3834975conv_block_8_3834977*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_8_layer_call_and_return_conditional_losses_3834105?
$conv_block_9/StatefulPartitionedCallStatefulPartitionedCall-conv_block_8/StatefulPartitionedCall:output:0conv_block_9_3834980conv_block_9_3834982conv_block_9_3834984conv_block_9_3834986conv_block_9_3834988*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_9_layer_call_and_return_conditional_losses_3834449?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall-conv_block_9/StatefulPartitionedCall:output:0conv2d_10_3834991*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_10_layer_call_and_return_conditional_losses_3834622?
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

?
I__inference_conv_block_6_layer_call_and_return_conditional_losses_3833505
input_10
sequential_10_3833493:@?$
sequential_10_3833495:	?$
sequential_10_3833497:	?$
sequential_10_3833499:	?$
sequential_10_3833501:	?
identity??%sequential_10/StatefulPartitionedCall?
%sequential_10/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_10_3833493sequential_10_3833495sequential_10_3833497sequential_10_3833499sequential_10_3833501*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_10_layer_call_and_return_conditional_losses_3833275?
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
?
?
J__inference_sequential_13_layer_call_and_return_conditional_losses_3836169

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
?(
?
I__inference_conv_block_9_layer_call_and_return_conditional_losses_3835806

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
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_3836353

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
?
I__inference_conv_block_8_layer_call_and_return_conditional_losses_3835710

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
?
?
.__inference_conv_block_6_layer_call_fn_3835540

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
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_6_layer_call_and_return_conditional_losses_3833462x
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

?
I__inference_conv_block_9_layer_call_and_return_conditional_losses_3834449

inputs0
sequential_13_3834437:?d#
sequential_13_3834439:d#
sequential_13_3834441:d#
sequential_13_3834443:d#
sequential_13_3834445:d
identity??%sequential_13/StatefulPartitionedCall?
%sequential_13/StatefulPartitionedCallStatefulPartitionedCallinputssequential_13_3834437sequential_13_3834439sequential_13_3834441sequential_13_3834443sequential_13_3834445*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_3834307?
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
?
?
E__inference_conv2d_9_layer_call_and_return_conditional_losses_3834286

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
?
?
+__inference_conv2d_10_layer_call_fn_3835813

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
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_10_layer_call_and_return_conditional_losses_3834622w
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
?
?
%__inference_signature_wrapper_3835436
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
*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__wrapped_model_3832832w
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
?

?
I__inference_conv_block_6_layer_call_and_return_conditional_losses_3833462

inputs0
sequential_10_3833450:@?$
sequential_10_3833452:	?$
sequential_10_3833454:	?$
sequential_10_3833456:	?$
sequential_10_3833458:	?
identity??%sequential_10/StatefulPartitionedCall?
%sequential_10/StatefulPartitionedCallStatefulPartitionedCallinputssequential_10_3833450sequential_10_3833452sequential_10_3833454sequential_10_3833456sequential_10_3833458*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_10_layer_call_and_return_conditional_losses_3833336?
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
?
?
J__inference_sequential_13_layer_call_and_return_conditional_losses_3834368

inputs+
conv2d_9_3834354:?d,
batch_normalization_13_3834357:d,
batch_normalization_13_3834359:d,
batch_normalization_13_3834361:d,
batch_normalization_13_3834363:d
identity??.batch_normalization_13/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_9_3834354*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_9_layer_call_and_return_conditional_losses_3834286?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0batch_normalization_13_3834357batch_normalization_13_3834359batch_normalization_13_3834361batch_normalization_13_3834363*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_3834261?
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
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_3834304}
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
?,
?

J__inference_discriminator_layer_call_and_return_conditional_losses_3834628

inputs.
conv_block_5_3834559:@"
conv_block_5_3834561:@"
conv_block_5_3834563:@"
conv_block_5_3834565:@"
conv_block_5_3834567:@/
conv_block_6_3834570:@?#
conv_block_6_3834572:	?#
conv_block_6_3834574:	?#
conv_block_6_3834576:	?#
conv_block_6_3834578:	?0
conv_block_7_3834581:??#
conv_block_7_3834583:	?#
conv_block_7_3834585:	?#
conv_block_7_3834587:	?#
conv_block_7_3834589:	?0
conv_block_8_3834592:??#
conv_block_8_3834594:	?#
conv_block_8_3834596:	?#
conv_block_8_3834598:	?#
conv_block_8_3834600:	?/
conv_block_9_3834603:?d"
conv_block_9_3834605:d"
conv_block_9_3834607:d"
conv_block_9_3834609:d"
conv_block_9_3834611:d+
conv2d_10_3834623:d
identity

identity_1??!conv2d_10/StatefulPartitionedCall?$conv_block_5/StatefulPartitionedCall?$conv_block_6/StatefulPartitionedCall?$conv_block_7/StatefulPartitionedCall?$conv_block_8/StatefulPartitionedCall?$conv_block_9/StatefulPartitionedCall?
$conv_block_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv_block_5_3834559conv_block_5_3834561conv_block_5_3834563conv_block_5_3834565conv_block_5_3834567*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_5_layer_call_and_return_conditional_losses_3833073?
$conv_block_6/StatefulPartitionedCallStatefulPartitionedCall-conv_block_5/StatefulPartitionedCall:output:0conv_block_6_3834570conv_block_6_3834572conv_block_6_3834574conv_block_6_3834576conv_block_6_3834578*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_6_layer_call_and_return_conditional_losses_3833417?
$conv_block_7/StatefulPartitionedCallStatefulPartitionedCall-conv_block_6/StatefulPartitionedCall:output:0conv_block_7_3834581conv_block_7_3834583conv_block_7_3834585conv_block_7_3834587conv_block_7_3834589*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_7_layer_call_and_return_conditional_losses_3833761?
$conv_block_8/StatefulPartitionedCallStatefulPartitionedCall-conv_block_7/StatefulPartitionedCall:output:0conv_block_8_3834592conv_block_8_3834594conv_block_8_3834596conv_block_8_3834598conv_block_8_3834600*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_8_layer_call_and_return_conditional_losses_3834105?
$conv_block_9/StatefulPartitionedCallStatefulPartitionedCall-conv_block_8/StatefulPartitionedCall:output:0conv_block_9_3834603conv_block_9_3834605conv_block_9_3834607conv_block_9_3834609conv_block_9_3834611*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_9_layer_call_and_return_conditional_losses_3834449?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall-conv_block_9/StatefulPartitionedCall:output:0conv2d_10_3834623*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_10_layer_call_and_return_conditional_losses_3834622?
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
?
?
I__inference_sequential_9_layer_call_and_return_conditional_losses_3833054
conv2d_5_input*
conv2d_5_3833040:@+
batch_normalization_9_3833043:@+
batch_normalization_9_3833045:@+
batch_normalization_9_3833047:@+
batch_normalization_9_3833049:@
identity??-batch_normalization_9/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputconv2d_5_3833040*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_3832910?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_9_3833043batch_normalization_9_3833045batch_normalization_9_3833047batch_normalization_9_3833049*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3832885?
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
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_3832928}
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
?(
?
I__inference_conv_block_6_layer_call_and_return_conditional_losses_3835584

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
?
?
/__inference_discriminator_layer_call_fn_3834685
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
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_discriminator_layer_call_and_return_conditional_losses_3834628w
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
?,
?

J__inference_discriminator_layer_call_and_return_conditional_losses_3835059
input_1.
conv_block_5_3834999:@"
conv_block_5_3835001:@"
conv_block_5_3835003:@"
conv_block_5_3835005:@"
conv_block_5_3835007:@/
conv_block_6_3835010:@?#
conv_block_6_3835012:	?#
conv_block_6_3835014:	?#
conv_block_6_3835016:	?#
conv_block_6_3835018:	?0
conv_block_7_3835021:??#
conv_block_7_3835023:	?#
conv_block_7_3835025:	?#
conv_block_7_3835027:	?#
conv_block_7_3835029:	?0
conv_block_8_3835032:??#
conv_block_8_3835034:	?#
conv_block_8_3835036:	?#
conv_block_8_3835038:	?#
conv_block_8_3835040:	?/
conv_block_9_3835043:?d"
conv_block_9_3835045:d"
conv_block_9_3835047:d"
conv_block_9_3835049:d"
conv_block_9_3835051:d+
conv2d_10_3835054:d
identity

identity_1??!conv2d_10/StatefulPartitionedCall?$conv_block_5/StatefulPartitionedCall?$conv_block_6/StatefulPartitionedCall?$conv_block_7/StatefulPartitionedCall?$conv_block_8/StatefulPartitionedCall?$conv_block_9/StatefulPartitionedCall?
$conv_block_5/StatefulPartitionedCallStatefulPartitionedCallinput_1conv_block_5_3834999conv_block_5_3835001conv_block_5_3835003conv_block_5_3835005conv_block_5_3835007*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_5_layer_call_and_return_conditional_losses_3833118?
$conv_block_6/StatefulPartitionedCallStatefulPartitionedCall-conv_block_5/StatefulPartitionedCall:output:0conv_block_6_3835010conv_block_6_3835012conv_block_6_3835014conv_block_6_3835016conv_block_6_3835018*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_6_layer_call_and_return_conditional_losses_3833462?
$conv_block_7/StatefulPartitionedCallStatefulPartitionedCall-conv_block_6/StatefulPartitionedCall:output:0conv_block_7_3835021conv_block_7_3835023conv_block_7_3835025conv_block_7_3835027conv_block_7_3835029*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_7_layer_call_and_return_conditional_losses_3833806?
$conv_block_8/StatefulPartitionedCallStatefulPartitionedCall-conv_block_7/StatefulPartitionedCall:output:0conv_block_8_3835032conv_block_8_3835034conv_block_8_3835036conv_block_8_3835038conv_block_8_3835040*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_8_layer_call_and_return_conditional_losses_3834150?
$conv_block_9/StatefulPartitionedCallStatefulPartitionedCall-conv_block_8/StatefulPartitionedCall:output:0conv_block_9_3835043conv_block_9_3835045conv_block_9_3835047conv_block_9_3835049conv_block_9_3835051*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_9_layer_call_and_return_conditional_losses_3834494?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall-conv_block_9/StatefulPartitionedCall:output:0conv2d_10_3835054*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_10_layer_call_and_return_conditional_losses_3834622?
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
.__inference_sequential_9_layer_call_fn_3833020
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
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_9_layer_call_and_return_conditional_losses_3832992w
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
?
?
J__inference_sequential_10_layer_call_and_return_conditional_losses_3833275

inputs+
conv2d_6_3833255:@?-
batch_normalization_10_3833258:	?-
batch_normalization_10_3833260:	?-
batch_normalization_10_3833262:	?-
batch_normalization_10_3833264:	?
identity??.batch_normalization_10/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_6_3833255*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_3833254?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_10_3833258batch_normalization_10_3833260batch_normalization_10_3833262batch_normalization_10_3833264*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_3833198?
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
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_3833272~
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
?
?
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_3836593

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
f
J__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_3836363

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
?
?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3832854

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
?	
?
8__inference_batch_normalization_12_layer_call_fn_3836476

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
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_3833886?
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
?
f
J__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_3833616

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
?
?
E__inference_conv2d_5_layer_call_and_return_conditional_losses_3836205

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
?;
?
 __inference__traced_save_3836723
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
?	
?
8__inference_batch_normalization_11_layer_call_fn_3836403

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
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_3833573?
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
?
?
F__inference_conv2d_10_layer_call_and_return_conditional_losses_3834622

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
?
?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_3833229

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
?
?
.__inference_conv_block_6_layer_call_fn_3833490
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
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_6_layer_call_and_return_conditional_losses_3833462x
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
?

?
I__inference_conv_block_7_layer_call_and_return_conditional_losses_3833864
input_11
sequential_11_3833852:??$
sequential_11_3833854:	?$
sequential_11_3833856:	?$
sequential_11_3833858:	?$
sequential_11_3833860:	?
identity??%sequential_11/StatefulPartitionedCall?
%sequential_11/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_11_3833852sequential_11_3833854sequential_11_3833856sequential_11_3833858sequential_11_3833860*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_11_layer_call_and_return_conditional_losses_3833680?
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
?
?
I__inference_sequential_9_layer_call_and_return_conditional_losses_3835873

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
?
?
J__inference_sequential_12_layer_call_and_return_conditional_losses_3836095

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
?
?
J__inference_sequential_12_layer_call_and_return_conditional_losses_3834069
conv2d_8_input,
conv2d_8_3834055:??-
batch_normalization_12_3834058:	?-
batch_normalization_12_3834060:	?-
batch_normalization_12_3834062:	?-
batch_normalization_12_3834064:	?
identity??.batch_normalization_12/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputconv2d_8_3834055*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_3833942?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_12_3834058batch_normalization_12_3834060batch_normalization_12_3834062batch_normalization_12_3834064*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_3833886?
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
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_3833960~
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
?(
?
I__inference_conv_block_8_layer_call_and_return_conditional_losses_3835732

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
?
f
J__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_3836449

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
?

?
I__inference_conv_block_8_layer_call_and_return_conditional_losses_3834208
input_11
sequential_12_3834196:??$
sequential_12_3834198:	?$
sequential_12_3834200:	?$
sequential_12_3834202:	?$
sequential_12_3834204:	?
identity??%sequential_12/StatefulPartitionedCall?
%sequential_12/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_12_3834196sequential_12_3834198sequential_12_3834200sequential_12_3834202sequential_12_3834204*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_12_layer_call_and_return_conditional_losses_3834024?
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
.__inference_conv_block_8_layer_call_fn_3834118
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
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_8_layer_call_and_return_conditional_losses_3834105x
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
?
?
J__inference_sequential_12_layer_call_and_return_conditional_losses_3833963

inputs,
conv2d_8_3833943:??-
batch_normalization_12_3833946:	?-
batch_normalization_12_3833948:	?-
batch_normalization_12_3833950:	?-
batch_normalization_12_3833952:	?
identity??.batch_normalization_12/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8_3833943*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_3833942?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_12_3833946batch_normalization_12_3833948batch_normalization_12_3833950batch_normalization_12_3833952*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_3833886?
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
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_3833960~
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
.__inference_conv_block_6_layer_call_fn_3835525

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
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_6_layer_call_and_return_conditional_losses_3833417x
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
?
?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3836267

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
J__inference_sequential_10_layer_call_and_return_conditional_losses_3833336

inputs+
conv2d_6_3833322:@?-
batch_normalization_10_3833325:	?-
batch_normalization_10_3833327:	?-
batch_normalization_10_3833329:	?-
batch_normalization_10_3833331:	?
identity??.batch_normalization_10/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_6_3833322*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_3833254?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_10_3833325batch_normalization_10_3833327batch_normalization_10_3833329batch_normalization_10_3833331*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_3833229?
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
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_3833272~
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
?
?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_3836439

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
*__inference_conv2d_8_layer_call_fn_3836456

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
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_3833942x
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
?
?
J__inference_sequential_12_layer_call_and_return_conditional_losses_3834024

inputs,
conv2d_8_3834010:??-
batch_normalization_12_3834013:	?-
batch_normalization_12_3834015:	?-
batch_normalization_12_3834017:	?-
batch_normalization_12_3834019:	?
identity??.batch_normalization_12/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8_3834010*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_3833942?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_12_3834013batch_normalization_12_3834015batch_normalization_12_3834017batch_normalization_12_3834019*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_3833917?
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
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_3833960~
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
.__inference_conv_block_9_layer_call_fn_3834522
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
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_9_layer_call_and_return_conditional_losses_3834494w
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
?'
?
I__inference_conv_block_5_layer_call_and_return_conditional_losses_3835510

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
?
?
J__inference_sequential_12_layer_call_and_return_conditional_losses_3834086
conv2d_8_input,
conv2d_8_3834072:??-
batch_normalization_12_3834075:	?-
batch_normalization_12_3834077:	?-
batch_normalization_12_3834079:	?-
batch_normalization_12_3834081:	?
identity??.batch_normalization_12/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallconv2d_8_inputconv2d_8_3834072*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_3833942?
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batch_normalization_12_3834075batch_normalization_12_3834077batch_normalization_12_3834079batch_normalization_12_3834081*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_3833917?
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
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_3833960~
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
I__inference_conv_block_9_layer_call_and_return_conditional_losses_3834537
input_10
sequential_13_3834525:?d#
sequential_13_3834527:d#
sequential_13_3834529:d#
sequential_13_3834531:d#
sequential_13_3834533:d
identity??%sequential_13/StatefulPartitionedCall?
%sequential_13/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_13_3834525sequential_13_3834527sequential_13_3834529sequential_13_3834531sequential_13_3834533*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_3834307?
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
?

?
I__inference_conv_block_6_layer_call_and_return_conditional_losses_3833520
input_10
sequential_10_3833508:@?$
sequential_10_3833510:	?$
sequential_10_3833512:	?$
sequential_10_3833514:	?$
sequential_10_3833516:	?
identity??%sequential_10/StatefulPartitionedCall?
%sequential_10/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_10_3833508sequential_10_3833510sequential_10_3833512sequential_10_3833514sequential_10_3833516*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_10_layer_call_and_return_conditional_losses_3833336?
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
?
f
J__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_3836535

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
?
?
E__inference_conv2d_6_layer_call_and_return_conditional_losses_3836291

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
K
/__inference_leaky_re_lu_9_layer_call_fn_3836616

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
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_3834304h
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
?
?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_3833542

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
?
?
.__inference_sequential_9_layer_call_fn_3835836

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
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_9_layer_call_and_return_conditional_losses_3832931w
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
?
?
J__inference_sequential_10_layer_call_and_return_conditional_losses_3835947

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
?
?
/__inference_sequential_12_layer_call_fn_3836073

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
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_12_layer_call_and_return_conditional_losses_3834024x
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
.__inference_conv_block_9_layer_call_fn_3834462
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
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_9_layer_call_and_return_conditional_losses_3834449w
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
.__inference_conv_block_5_layer_call_fn_3833086
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
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_5_layer_call_and_return_conditional_losses_3833073w
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
.__inference_conv_block_7_layer_call_fn_3835614

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
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_7_layer_call_and_return_conditional_losses_3833806x
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
?
?
J__inference_sequential_10_layer_call_and_return_conditional_losses_3833398
conv2d_6_input+
conv2d_6_3833384:@?-
batch_normalization_10_3833387:	?-
batch_normalization_10_3833389:	?-
batch_normalization_10_3833391:	?-
batch_normalization_10_3833393:	?
identity??.batch_normalization_10/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallconv2d_6_inputconv2d_6_3833384*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_3833254?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_10_3833387batch_normalization_10_3833389batch_normalization_10_3833391batch_normalization_10_3833393*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_3833229?
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
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_3833272~
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
?
?
/__inference_discriminator_layer_call_fn_3835118

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
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_discriminator_layer_call_and_return_conditional_losses_3834628w
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
?
K
/__inference_leaky_re_lu_7_layer_call_fn_3836444

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
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_3833616i
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
?,
?

J__inference_discriminator_layer_call_and_return_conditional_losses_3834817

inputs.
conv_block_5_3834757:@"
conv_block_5_3834759:@"
conv_block_5_3834761:@"
conv_block_5_3834763:@"
conv_block_5_3834765:@/
conv_block_6_3834768:@?#
conv_block_6_3834770:	?#
conv_block_6_3834772:	?#
conv_block_6_3834774:	?#
conv_block_6_3834776:	?0
conv_block_7_3834779:??#
conv_block_7_3834781:	?#
conv_block_7_3834783:	?#
conv_block_7_3834785:	?#
conv_block_7_3834787:	?0
conv_block_8_3834790:??#
conv_block_8_3834792:	?#
conv_block_8_3834794:	?#
conv_block_8_3834796:	?#
conv_block_8_3834798:	?/
conv_block_9_3834801:?d"
conv_block_9_3834803:d"
conv_block_9_3834805:d"
conv_block_9_3834807:d"
conv_block_9_3834809:d+
conv2d_10_3834812:d
identity

identity_1??!conv2d_10/StatefulPartitionedCall?$conv_block_5/StatefulPartitionedCall?$conv_block_6/StatefulPartitionedCall?$conv_block_7/StatefulPartitionedCall?$conv_block_8/StatefulPartitionedCall?$conv_block_9/StatefulPartitionedCall?
$conv_block_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv_block_5_3834757conv_block_5_3834759conv_block_5_3834761conv_block_5_3834763conv_block_5_3834765*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_5_layer_call_and_return_conditional_losses_3833118?
$conv_block_6/StatefulPartitionedCallStatefulPartitionedCall-conv_block_5/StatefulPartitionedCall:output:0conv_block_6_3834768conv_block_6_3834770conv_block_6_3834772conv_block_6_3834774conv_block_6_3834776*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_6_layer_call_and_return_conditional_losses_3833462?
$conv_block_7/StatefulPartitionedCallStatefulPartitionedCall-conv_block_6/StatefulPartitionedCall:output:0conv_block_7_3834779conv_block_7_3834781conv_block_7_3834783conv_block_7_3834785conv_block_7_3834787*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_7_layer_call_and_return_conditional_losses_3833806?
$conv_block_8/StatefulPartitionedCallStatefulPartitionedCall-conv_block_7/StatefulPartitionedCall:output:0conv_block_8_3834790conv_block_8_3834792conv_block_8_3834794conv_block_8_3834796conv_block_8_3834798*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_8_layer_call_and_return_conditional_losses_3834150?
$conv_block_9/StatefulPartitionedCallStatefulPartitionedCall-conv_block_8/StatefulPartitionedCall:output:0conv_block_9_3834801conv_block_9_3834803conv_block_9_3834805conv_block_9_3834807conv_block_9_3834809*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_9_layer_call_and_return_conditional_losses_3834494?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall-conv_block_9/StatefulPartitionedCall:output:0conv2d_10_3834812*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_10_layer_call_and_return_conditional_losses_3834622?
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
.__inference_sequential_9_layer_call_fn_3832944
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
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_9_layer_call_and_return_conditional_losses_3832931w
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
?
?
J__inference_sequential_11_layer_call_and_return_conditional_losses_3836021

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
?!
?
J__inference_sequential_10_layer_call_and_return_conditional_losses_3835969

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
?
?
E__inference_conv2d_7_layer_call_and_return_conditional_losses_3833598

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

?
I__inference_conv_block_5_layer_call_and_return_conditional_losses_3833176
input_1.
sequential_9_3833164:@"
sequential_9_3833166:@"
sequential_9_3833168:@"
sequential_9_3833170:@"
sequential_9_3833172:@
identity??$sequential_9/StatefulPartitionedCall?
$sequential_9/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_9_3833164sequential_9_3833166sequential_9_3833168sequential_9_3833170sequential_9_3833172*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_9_layer_call_and_return_conditional_losses_3832992?
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
?

?
I__inference_conv_block_5_layer_call_and_return_conditional_losses_3833073

inputs.
sequential_9_3833061:@"
sequential_9_3833063:@"
sequential_9_3833065:@"
sequential_9_3833067:@"
sequential_9_3833069:@
identity??$sequential_9/StatefulPartitionedCall?
$sequential_9/StatefulPartitionedCallStatefulPartitionedCallinputssequential_9_3833061sequential_9_3833063sequential_9_3833065sequential_9_3833067sequential_9_3833069*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_9_layer_call_and_return_conditional_losses_3832931?
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
?

?
I__inference_conv_block_7_layer_call_and_return_conditional_losses_3833849
input_11
sequential_11_3833837:??$
sequential_11_3833839:	?$
sequential_11_3833841:	?$
sequential_11_3833843:	?$
sequential_11_3833845:	?
identity??%sequential_11/StatefulPartitionedCall?
%sequential_11/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_11_3833837sequential_11_3833839sequential_11_3833841sequential_11_3833843sequential_11_3833845*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_11_layer_call_and_return_conditional_losses_3833619?
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
?	
?
7__inference_batch_normalization_9_layer_call_fn_3836218

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
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3832854?
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
?
f
J__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_3833960

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
I__inference_conv_block_5_layer_call_and_return_conditional_losses_3833161
input_1.
sequential_9_3833149:@"
sequential_9_3833151:@"
sequential_9_3833153:@"
sequential_9_3833155:@"
sequential_9_3833157:@
identity??$sequential_9/StatefulPartitionedCall?
$sequential_9/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_9_3833149sequential_9_3833151sequential_9_3833153sequential_9_3833155sequential_9_3833157*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_sequential_9_layer_call_and_return_conditional_losses_3832931?
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
?	
?
/__inference_sequential_12_layer_call_fn_3834052
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
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_12_layer_call_and_return_conditional_losses_3834024x
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
.__inference_conv_block_8_layer_call_fn_3835688

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
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_8_layer_call_and_return_conditional_losses_3834150x
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
?	
?
/__inference_sequential_10_layer_call_fn_3833288
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
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_10_layer_call_and_return_conditional_losses_3833275x
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
?
?
I__inference_conv_block_9_layer_call_and_return_conditional_losses_3835784

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
?	
?
/__inference_sequential_11_layer_call_fn_3833632
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
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_11_layer_call_and_return_conditional_losses_3833619x
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
??
?(
J__inference_discriminator_layer_call_and_return_conditional_losses_3835375

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
?	
?
7__inference_batch_normalization_9_layer_call_fn_3836231

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
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3832885?
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
?
?
*__inference_conv2d_7_layer_call_fn_3836370

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
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_3833598x
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
?
?
.__inference_conv_block_5_layer_call_fn_3835451

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
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_5_layer_call_and_return_conditional_losses_3833073w
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
?
I__inference_sequential_9_layer_call_and_return_conditional_losses_3835895

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
?	
?
8__inference_batch_normalization_10_layer_call_fn_3836317

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
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_3833229?
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
?
/__inference_sequential_10_layer_call_fn_3833364
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
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_10_layer_call_and_return_conditional_losses_3833336x
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
/__inference_sequential_10_layer_call_fn_3835910

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
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_10_layer_call_and_return_conditional_losses_3833275x
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
.__inference_conv_block_9_layer_call_fn_3835747

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
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_9_layer_call_and_return_conditional_losses_3834449w
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
J__inference_sequential_13_layer_call_and_return_conditional_losses_3834413
conv2d_9_input+
conv2d_9_3834399:?d,
batch_normalization_13_3834402:d,
batch_normalization_13_3834404:d,
batch_normalization_13_3834406:d,
batch_normalization_13_3834408:d
identity??.batch_normalization_13/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCallconv2d_9_inputconv2d_9_3834399*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_9_layer_call_and_return_conditional_losses_3834286?
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0batch_normalization_13_3834402batch_normalization_13_3834404batch_normalization_13_3834406batch_normalization_13_3834408*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_3834230?
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
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_3834304}
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
?!
?
J__inference_sequential_12_layer_call_and_return_conditional_losses_3836117

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
?

?
I__inference_conv_block_8_layer_call_and_return_conditional_losses_3834150

inputs1
sequential_12_3834138:??$
sequential_12_3834140:	?$
sequential_12_3834142:	?$
sequential_12_3834144:	?$
sequential_12_3834146:	?
identity??%sequential_12/StatefulPartitionedCall?
%sequential_12/StatefulPartitionedCallStatefulPartitionedCallinputssequential_12_3834138sequential_12_3834140sequential_12_3834142sequential_12_3834144sequential_12_3834146*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_12_layer_call_and_return_conditional_losses_3834024?
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
?

?
I__inference_conv_block_9_layer_call_and_return_conditional_losses_3834494

inputs0
sequential_13_3834482:?d#
sequential_13_3834484:d#
sequential_13_3834486:d#
sequential_13_3834488:d#
sequential_13_3834490:d
identity??%sequential_13/StatefulPartitionedCall?
%sequential_13/StatefulPartitionedCallStatefulPartitionedCallinputssequential_13_3834482sequential_13_3834484sequential_13_3834486sequential_13_3834488sequential_13_3834490*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_3834368?
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
?!
?
J__inference_sequential_11_layer_call_and_return_conditional_losses_3836043

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
?
?
.__inference_conv_block_5_layer_call_fn_3835466

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
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_5_layer_call_and_return_conditional_losses_3833118w
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
?
8__inference_batch_normalization_11_layer_call_fn_3836390

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
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_3833542?
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
?
?
J__inference_sequential_11_layer_call_and_return_conditional_losses_3833680

inputs,
conv2d_7_3833666:??-
batch_normalization_11_3833669:	?-
batch_normalization_11_3833671:	?-
batch_normalization_11_3833673:	?-
batch_normalization_11_3833675:	?
identity??.batch_normalization_11/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_7_3833666*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_3833598?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_11_3833669batch_normalization_11_3833671batch_normalization_11_3833673batch_normalization_11_3833675*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_3833573?
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
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_3833616~
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
?
/__inference_sequential_13_layer_call_fn_3834396
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
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_3834368w
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
?	
?
8__inference_batch_normalization_13_layer_call_fn_3836562

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
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_3834230?
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
E__inference_conv2d_9_layer_call_and_return_conditional_losses_3836549

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
?
f
J__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_3832928

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
?
?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_3833573

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
?
?
I__inference_sequential_9_layer_call_and_return_conditional_losses_3832931

inputs*
conv2d_5_3832911:@+
batch_normalization_9_3832914:@+
batch_normalization_9_3832916:@+
batch_normalization_9_3832918:@+
batch_normalization_9_3832920:@
identity??-batch_normalization_9/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_3832911*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_3832910?
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_9_3832914batch_normalization_9_3832916batch_normalization_9_3832918batch_normalization_9_3832920*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3832854?
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
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_3832928}
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
.__inference_conv_block_7_layer_call_fn_3835599

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
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_7_layer_call_and_return_conditional_losses_3833761x
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
?
f
J__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_3836621

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
E__inference_conv2d_8_layer_call_and_return_conditional_losses_3833942

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
?
?
.__inference_conv_block_9_layer_call_fn_3835762

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
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_9_layer_call_and_return_conditional_losses_3834494w
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
?
?
.__inference_conv_block_7_layer_call_fn_3833834
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
*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_conv_block_7_layer_call_and_return_conditional_losses_3833806x
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

?
I__inference_conv_block_7_layer_call_and_return_conditional_losses_3833761

inputs1
sequential_11_3833749:??$
sequential_11_3833751:	?$
sequential_11_3833753:	?$
sequential_11_3833755:	?$
sequential_11_3833757:	?
identity??%sequential_11/StatefulPartitionedCall?
%sequential_11/StatefulPartitionedCallStatefulPartitionedCallinputssequential_11_3833749sequential_11_3833751sequential_11_3833753sequential_11_3833755sequential_11_3833757*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_11_layer_call_and_return_conditional_losses_3833619?
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
?
?
/__inference_sequential_12_layer_call_fn_3836058

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
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_12_layer_call_and_return_conditional_losses_3833963x
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
?
?
J__inference_sequential_11_layer_call_and_return_conditional_losses_3833742
conv2d_7_input,
conv2d_7_3833728:??-
batch_normalization_11_3833731:	?-
batch_normalization_11_3833733:	?-
batch_normalization_11_3833735:	?-
batch_normalization_11_3833737:	?
identity??.batch_normalization_11/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCallconv2d_7_inputconv2d_7_3833728*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_3833598?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_11_3833731batch_normalization_11_3833733batch_normalization_11_3833735batch_normalization_11_3833737*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_3833573?
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
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_3833616~
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
?
?
J__inference_sequential_11_layer_call_and_return_conditional_losses_3833619

inputs,
conv2d_7_3833599:??-
batch_normalization_11_3833602:	?-
batch_normalization_11_3833604:	?-
batch_normalization_11_3833606:	?-
batch_normalization_11_3833608:	?
identity??.batch_normalization_11/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_7_3833599*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_3833598?
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_11_3833602batch_normalization_11_3833604batch_normalization_11_3833606batch_normalization_11_3833608*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_3833542?
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
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_3833616~
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
?
I__inference_conv_block_5_layer_call_and_return_conditional_losses_3835488

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
?(
?
I__inference_conv_block_7_layer_call_and_return_conditional_losses_3835658

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
?
?
F__inference_conv2d_10_layer_call_and_return_conditional_losses_3835821

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
?
?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_3836335

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
/__inference_sequential_13_layer_call_fn_3836147

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
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_3834368w
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
E__inference_conv2d_5_layer_call_and_return_conditional_losses_3832910

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
J__inference_sequential_10_layer_call_and_return_conditional_losses_3833381
conv2d_6_input+
conv2d_6_3833367:@?-
batch_normalization_10_3833370:	?-
batch_normalization_10_3833372:	?-
batch_normalization_10_3833374:	?-
batch_normalization_10_3833376:	?
identity??.batch_normalization_10/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallconv2d_6_inputconv2d_6_3833367*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_3833254?
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_10_3833370batch_normalization_10_3833372batch_normalization_10_3833374batch_normalization_10_3833376*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_3833198?
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
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_3833272~
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
?
?
*__inference_conv2d_9_layer_call_fn_3836542

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
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_9_layer_call_and_return_conditional_losses_3834286w
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
?
K
/__inference_leaky_re_lu_5_layer_call_fn_3836272

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
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_3832928h
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
?

?
I__inference_conv_block_7_layer_call_and_return_conditional_losses_3833806

inputs1
sequential_11_3833794:??$
sequential_11_3833796:	?$
sequential_11_3833798:	?$
sequential_11_3833800:	?$
sequential_11_3833802:	?
identity??%sequential_11/StatefulPartitionedCall?
%sequential_11/StatefulPartitionedCallStatefulPartitionedCallinputssequential_11_3833794sequential_11_3833796sequential_11_3833798sequential_11_3833800sequential_11_3833802*
Tin

2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_11_layer_call_and_return_conditional_losses_3833680?
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
?
?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_3833198

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
?
?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_3836507

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
?

?
I__inference_conv_block_9_layer_call_and_return_conditional_losses_3834552
input_10
sequential_13_3834540:?d#
sequential_13_3834542:d#
sequential_13_3834544:d#
sequential_13_3834546:d#
sequential_13_3834548:d
identity??%sequential_13/StatefulPartitionedCall?
%sequential_13/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_13_3834540sequential_13_3834542sequential_13_3834544sequential_13_3834546sequential_13_3834548*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????d*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_3834368?
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
?
?
*__inference_conv2d_6_layer_call_fn_3836284

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
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_3833254x
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
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_model
?

conv_layer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_model
?

conv_layer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_model
?

conv_layer
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_model
?
$
conv_layer
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses"
_tf_keras_model
?
+
conv_layer
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses"
_tf_keras_model
?

2kernel
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses"
_tf_keras_layer
?
90
:1
;2
<3
=4
>5
?6
@7
A8
B9
C10
D11
E12
F13
G14
H15
I16
J17
K18
L19
M20
N21
O22
P23
Q24
225"
trackable_list_wrapper
?
90
:1
;2
>3
?4
@5
C6
D7
E8
H9
I10
J11
M12
N13
O14
215"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_discriminator_layer_call_fn_3834685
/__inference_discriminator_layer_call_fn_3835118
/__inference_discriminator_layer_call_fn_3835177
/__inference_discriminator_layer_call_fn_3834933?
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
J__inference_discriminator_layer_call_and_return_conditional_losses_3835276
J__inference_discriminator_layer_call_and_return_conditional_losses_3835375
J__inference_discriminator_layer_call_and_return_conditional_losses_3834996
J__inference_discriminator_layer_call_and_return_conditional_losses_3835059?
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
"__inference__wrapped_model_3832832input_1"?
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
,
Wserving_default"
signature_map
?
Xlayer_with_weights-0
Xlayer-0
Ylayer_with_weights-1
Ylayer-1
Zlayer-2
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses"
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
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
.__inference_conv_block_5_layer_call_fn_3833086
.__inference_conv_block_5_layer_call_fn_3835451
.__inference_conv_block_5_layer_call_fn_3835466
.__inference_conv_block_5_layer_call_fn_3833146?
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
I__inference_conv_block_5_layer_call_and_return_conditional_losses_3835488
I__inference_conv_block_5_layer_call_and_return_conditional_losses_3835510
I__inference_conv_block_5_layer_call_and_return_conditional_losses_3833161
I__inference_conv_block_5_layer_call_and_return_conditional_losses_3833176?
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
?
flayer_with_weights-0
flayer-0
glayer_with_weights-1
glayer-1
hlayer-2
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses"
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
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
.__inference_conv_block_6_layer_call_fn_3833430
.__inference_conv_block_6_layer_call_fn_3835525
.__inference_conv_block_6_layer_call_fn_3835540
.__inference_conv_block_6_layer_call_fn_3833490?
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
I__inference_conv_block_6_layer_call_and_return_conditional_losses_3835562
I__inference_conv_block_6_layer_call_and_return_conditional_losses_3835584
I__inference_conv_block_6_layer_call_and_return_conditional_losses_3833505
I__inference_conv_block_6_layer_call_and_return_conditional_losses_3833520?
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
?
tlayer_with_weights-0
tlayer-0
ulayer_with_weights-1
ulayer-1
vlayer-2
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses"
_tf_keras_sequential
C
C0
D1
E2
F3
G4"
trackable_list_wrapper
5
C0
D1
E2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
}non_trainable_variables

~layers
metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
?2?
.__inference_conv_block_7_layer_call_fn_3833774
.__inference_conv_block_7_layer_call_fn_3835599
.__inference_conv_block_7_layer_call_fn_3835614
.__inference_conv_block_7_layer_call_fn_3833834?
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
I__inference_conv_block_7_layer_call_and_return_conditional_losses_3835636
I__inference_conv_block_7_layer_call_and_return_conditional_losses_3835658
I__inference_conv_block_7_layer_call_and_return_conditional_losses_3833849
I__inference_conv_block_7_layer_call_and_return_conditional_losses_3833864?
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
?
?layer_with_weights-0
?layer-0
?layer_with_weights-1
?layer-1
?layer-2
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
C
H0
I1
J2
K3
L4"
trackable_list_wrapper
5
H0
I1
J2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
?2?
.__inference_conv_block_8_layer_call_fn_3834118
.__inference_conv_block_8_layer_call_fn_3835673
.__inference_conv_block_8_layer_call_fn_3835688
.__inference_conv_block_8_layer_call_fn_3834178?
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
I__inference_conv_block_8_layer_call_and_return_conditional_losses_3835710
I__inference_conv_block_8_layer_call_and_return_conditional_losses_3835732
I__inference_conv_block_8_layer_call_and_return_conditional_losses_3834193
I__inference_conv_block_8_layer_call_and_return_conditional_losses_3834208?
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
?
?layer_with_weights-0
?layer-0
?layer_with_weights-1
?layer-1
?layer-2
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
C
M0
N1
O2
P3
Q4"
trackable_list_wrapper
5
M0
N1
O2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
?2?
.__inference_conv_block_9_layer_call_fn_3834462
.__inference_conv_block_9_layer_call_fn_3835747
.__inference_conv_block_9_layer_call_fn_3835762
.__inference_conv_block_9_layer_call_fn_3834522?
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
I__inference_conv_block_9_layer_call_and_return_conditional_losses_3835784
I__inference_conv_block_9_layer_call_and_return_conditional_losses_3835806
I__inference_conv_block_9_layer_call_and_return_conditional_losses_3834537
I__inference_conv_block_9_layer_call_and_return_conditional_losses_3834552?
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
8:6d2discriminator/conv2d_10/kernel
'
20"
trackable_list_wrapper
'
20"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_conv2d_10_layer_call_fn_3835813?
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
F__inference_conv2d_10_layer_call_and_return_conditional_losses_3835821?
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
<0
=1
A2
B3
F4
G5
K6
L7
P8
Q9"
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
?B?
%__inference_signature_wrapper_3835436input_1"?
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
?

9kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
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
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
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
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
?2?
.__inference_sequential_9_layer_call_fn_3832944
.__inference_sequential_9_layer_call_fn_3835836
.__inference_sequential_9_layer_call_fn_3835851
.__inference_sequential_9_layer_call_fn_3833020?
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
I__inference_sequential_9_layer_call_and_return_conditional_losses_3835873
I__inference_sequential_9_layer_call_and_return_conditional_losses_3835895
I__inference_sequential_9_layer_call_and_return_conditional_losses_3833037
I__inference_sequential_9_layer_call_and_return_conditional_losses_3833054?
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
.
<0
=1"
trackable_list_wrapper
'
0"
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
?__call__
+?&call_and_return_all_conditional_losses"
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
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
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
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_sequential_10_layer_call_fn_3833288
/__inference_sequential_10_layer_call_fn_3835910
/__inference_sequential_10_layer_call_fn_3835925
/__inference_sequential_10_layer_call_fn_3833364?
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
J__inference_sequential_10_layer_call_and_return_conditional_losses_3835947
J__inference_sequential_10_layer_call_and_return_conditional_losses_3835969
J__inference_sequential_10_layer_call_and_return_conditional_losses_3833381
J__inference_sequential_10_layer_call_and_return_conditional_losses_3833398?
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
.
A0
B1"
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

Ckernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	Dgamma
Ebeta
Fmoving_mean
Gmoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
C
C0
D1
E2
F3
G4"
trackable_list_wrapper
5
C0
D1
E2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_sequential_11_layer_call_fn_3833632
/__inference_sequential_11_layer_call_fn_3835984
/__inference_sequential_11_layer_call_fn_3835999
/__inference_sequential_11_layer_call_fn_3833708?
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
J__inference_sequential_11_layer_call_and_return_conditional_losses_3836021
J__inference_sequential_11_layer_call_and_return_conditional_losses_3836043
J__inference_sequential_11_layer_call_and_return_conditional_losses_3833725
J__inference_sequential_11_layer_call_and_return_conditional_losses_3833742?
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
.
F0
G1"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

Hkernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	Igamma
Jbeta
Kmoving_mean
Lmoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
C
H0
I1
J2
K3
L4"
trackable_list_wrapper
5
H0
I1
J2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_sequential_12_layer_call_fn_3833976
/__inference_sequential_12_layer_call_fn_3836058
/__inference_sequential_12_layer_call_fn_3836073
/__inference_sequential_12_layer_call_fn_3834052?
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
J__inference_sequential_12_layer_call_and_return_conditional_losses_3836095
J__inference_sequential_12_layer_call_and_return_conditional_losses_3836117
J__inference_sequential_12_layer_call_and_return_conditional_losses_3834069
J__inference_sequential_12_layer_call_and_return_conditional_losses_3834086?
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
.
K0
L1"
trackable_list_wrapper
'
$0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

Mkernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	Ngamma
Obeta
Pmoving_mean
Qmoving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
C
M0
N1
O2
P3
Q4"
trackable_list_wrapper
5
M0
N1
O2"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_sequential_13_layer_call_fn_3834320
/__inference_sequential_13_layer_call_fn_3836132
/__inference_sequential_13_layer_call_fn_3836147
/__inference_sequential_13_layer_call_fn_3834396?
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
J__inference_sequential_13_layer_call_and_return_conditional_losses_3836169
J__inference_sequential_13_layer_call_and_return_conditional_losses_3836191
J__inference_sequential_13_layer_call_and_return_conditional_losses_3834413
J__inference_sequential_13_layer_call_and_return_conditional_losses_3834430?
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
.
P0
Q1"
trackable_list_wrapper
'
+0"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_conv2d_5_layer_call_fn_3836198?
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
E__inference_conv2d_5_layer_call_and_return_conditional_losses_3836205?
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
7__inference_batch_normalization_9_layer_call_fn_3836218
7__inference_batch_normalization_9_layer_call_fn_3836231?
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
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3836249
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3836267?
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_leaky_re_lu_5_layer_call_fn_3836272?
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
J__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_3836277?
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
.
<0
=1"
trackable_list_wrapper
5
X0
Y1
Z2"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_conv2d_6_layer_call_fn_3836284?
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
E__inference_conv2d_6_layer_call_and_return_conditional_losses_3836291?
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
8__inference_batch_normalization_10_layer_call_fn_3836304
8__inference_batch_normalization_10_layer_call_fn_3836317?
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
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_3836335
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_3836353?
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_leaky_re_lu_6_layer_call_fn_3836358?
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
J__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_3836363?
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
.
A0
B1"
trackable_list_wrapper
5
f0
g1
h2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
C0"
trackable_list_wrapper
'
C0"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_conv2d_7_layer_call_fn_3836370?
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
E__inference_conv2d_7_layer_call_and_return_conditional_losses_3836377?
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
 "
trackable_list_wrapper
<
D0
E1
F2
G3"
trackable_list_wrapper
.
D0
E1"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
8__inference_batch_normalization_11_layer_call_fn_3836390
8__inference_batch_normalization_11_layer_call_fn_3836403?
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
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_3836421
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_3836439?
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_leaky_re_lu_7_layer_call_fn_3836444?
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
J__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_3836449?
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
.
F0
G1"
trackable_list_wrapper
5
t0
u1
v2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
H0"
trackable_list_wrapper
'
H0"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_conv2d_8_layer_call_fn_3836456?
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
E__inference_conv2d_8_layer_call_and_return_conditional_losses_3836463?
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
 "
trackable_list_wrapper
<
I0
J1
K2
L3"
trackable_list_wrapper
.
I0
J1"
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
8__inference_batch_normalization_12_layer_call_fn_3836476
8__inference_batch_normalization_12_layer_call_fn_3836489?
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
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_3836507
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_3836525?
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
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_leaky_re_lu_8_layer_call_fn_3836530?
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
J__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_3836535?
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
.
K0
L1"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
M0"
trackable_list_wrapper
'
M0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_conv2d_9_layer_call_fn_3836542?
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
E__inference_conv2d_9_layer_call_and_return_conditional_losses_3836549?
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
 "
trackable_list_wrapper
<
N0
O1
P2
Q3"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
8__inference_batch_normalization_13_layer_call_fn_3836562
8__inference_batch_normalization_13_layer_call_fn_3836575?
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
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_3836593
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_3836611?
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
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_leaky_re_lu_9_layer_call_fn_3836616?
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
J__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_3836621?
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
.
P0
Q1"
trackable_list_wrapper
8
?0
?1
?2"
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
F0
G1"
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
K0
L1"
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
P0
Q1"
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
trackable_dict_wrapper?
"__inference__wrapped_model_3832832?9:;<=>?@ABCDEFGHIJKLMNOPQ2:?7
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
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_3836335??@ABN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
S__inference_batch_normalization_10_layer_call_and_return_conditional_losses_3836353??@ABN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
8__inference_batch_normalization_10_layer_call_fn_3836304??@ABN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
8__inference_batch_normalization_10_layer_call_fn_3836317??@ABN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_3836421?DEFGN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
S__inference_batch_normalization_11_layer_call_and_return_conditional_losses_3836439?DEFGN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
8__inference_batch_normalization_11_layer_call_fn_3836390?DEFGN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
8__inference_batch_normalization_11_layer_call_fn_3836403?DEFGN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_3836507?IJKLN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_3836525?IJKLN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
8__inference_batch_normalization_12_layer_call_fn_3836476?IJKLN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
8__inference_batch_normalization_12_layer_call_fn_3836489?IJKLN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_3836593?NOPQM?J
C?@
:?7
inputs+???????????????????????????d
p 
? "??<
5?2
0+???????????????????????????d
? ?
S__inference_batch_normalization_13_layer_call_and_return_conditional_losses_3836611?NOPQM?J
C?@
:?7
inputs+???????????????????????????d
p
? "??<
5?2
0+???????????????????????????d
? ?
8__inference_batch_normalization_13_layer_call_fn_3836562?NOPQM?J
C?@
:?7
inputs+???????????????????????????d
p 
? "2?/+???????????????????????????d?
8__inference_batch_normalization_13_layer_call_fn_3836575?NOPQM?J
C?@
:?7
inputs+???????????????????????????d
p
? "2?/+???????????????????????????d?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3836249?:;<=M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
R__inference_batch_normalization_9_layer_call_and_return_conditional_losses_3836267?:;<=M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
7__inference_batch_normalization_9_layer_call_fn_3836218?:;<=M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
7__inference_batch_normalization_9_layer_call_fn_3836231?:;<=M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
F__inference_conv2d_10_layer_call_and_return_conditional_losses_3835821k27?4
-?*
(?%
inputs?????????d
? "-?*
#? 
0?????????
? ?
+__inference_conv2d_10_layer_call_fn_3835813^27?4
-?*
(?%
inputs?????????d
? " ???????????
E__inference_conv2d_5_layer_call_and_return_conditional_losses_3836205m99?6
/?,
*?'
inputs???????????
? "-?*
#? 
0?????????@@@
? ?
*__inference_conv2d_5_layer_call_fn_3836198`99?6
/?,
*?'
inputs???????????
? " ??????????@@@?
E__inference_conv2d_6_layer_call_and_return_conditional_losses_3836291l>7?4
-?*
(?%
inputs?????????@@@
? ".?+
$?!
0?????????  ?
? ?
*__inference_conv2d_6_layer_call_fn_3836284_>7?4
-?*
(?%
inputs?????????@@@
? "!??????????  ??
E__inference_conv2d_7_layer_call_and_return_conditional_losses_3836377mC8?5
.?+
)?&
inputs?????????  ?
? ".?+
$?!
0??????????
? ?
*__inference_conv2d_7_layer_call_fn_3836370`C8?5
.?+
)?&
inputs?????????  ?
? "!????????????
E__inference_conv2d_8_layer_call_and_return_conditional_losses_3836463mH8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
*__inference_conv2d_8_layer_call_fn_3836456`H8?5
.?+
)?&
inputs??????????
? "!????????????
E__inference_conv2d_9_layer_call_and_return_conditional_losses_3836549lM8?5
.?+
)?&
inputs??????????
? "-?*
#? 
0?????????d
? ?
*__inference_conv2d_9_layer_call_fn_3836542_M8?5
.?+
)?&
inputs??????????
? " ??????????d?
I__inference_conv_block_5_layer_call_and_return_conditional_losses_3833161v9:;<=>?;
4?1
+?(
input_1???????????
p 
? "-?*
#? 
0?????????@@@
? ?
I__inference_conv_block_5_layer_call_and_return_conditional_losses_3833176v9:;<=>?;
4?1
+?(
input_1???????????
p
? "-?*
#? 
0?????????@@@
? ?
I__inference_conv_block_5_layer_call_and_return_conditional_losses_3835488u9:;<==?:
3?0
*?'
inputs???????????
p 
? "-?*
#? 
0?????????@@@
? ?
I__inference_conv_block_5_layer_call_and_return_conditional_losses_3835510u9:;<==?:
3?0
*?'
inputs???????????
p
? "-?*
#? 
0?????????@@@
? ?
.__inference_conv_block_5_layer_call_fn_3833086i9:;<=>?;
4?1
+?(
input_1???????????
p 
? " ??????????@@@?
.__inference_conv_block_5_layer_call_fn_3833146i9:;<=>?;
4?1
+?(
input_1???????????
p
? " ??????????@@@?
.__inference_conv_block_5_layer_call_fn_3835451h9:;<==?:
3?0
*?'
inputs???????????
p 
? " ??????????@@@?
.__inference_conv_block_5_layer_call_fn_3835466h9:;<==?:
3?0
*?'
inputs???????????
p
? " ??????????@@@?
I__inference_conv_block_6_layer_call_and_return_conditional_losses_3833505u>?@AB<?9
2?/
)?&
input_1?????????@@@
p 
? ".?+
$?!
0?????????  ?
? ?
I__inference_conv_block_6_layer_call_and_return_conditional_losses_3833520u>?@AB<?9
2?/
)?&
input_1?????????@@@
p
? ".?+
$?!
0?????????  ?
? ?
I__inference_conv_block_6_layer_call_and_return_conditional_losses_3835562t>?@AB;?8
1?.
(?%
inputs?????????@@@
p 
? ".?+
$?!
0?????????  ?
? ?
I__inference_conv_block_6_layer_call_and_return_conditional_losses_3835584t>?@AB;?8
1?.
(?%
inputs?????????@@@
p
? ".?+
$?!
0?????????  ?
? ?
.__inference_conv_block_6_layer_call_fn_3833430h>?@AB<?9
2?/
)?&
input_1?????????@@@
p 
? "!??????????  ??
.__inference_conv_block_6_layer_call_fn_3833490h>?@AB<?9
2?/
)?&
input_1?????????@@@
p
? "!??????????  ??
.__inference_conv_block_6_layer_call_fn_3835525g>?@AB;?8
1?.
(?%
inputs?????????@@@
p 
? "!??????????  ??
.__inference_conv_block_6_layer_call_fn_3835540g>?@AB;?8
1?.
(?%
inputs?????????@@@
p
? "!??????????  ??
I__inference_conv_block_7_layer_call_and_return_conditional_losses_3833849vCDEFG=?:
3?0
*?'
input_1?????????  ?
p 
? ".?+
$?!
0??????????
? ?
I__inference_conv_block_7_layer_call_and_return_conditional_losses_3833864vCDEFG=?:
3?0
*?'
input_1?????????  ?
p
? ".?+
$?!
0??????????
? ?
I__inference_conv_block_7_layer_call_and_return_conditional_losses_3835636uCDEFG<?9
2?/
)?&
inputs?????????  ?
p 
? ".?+
$?!
0??????????
? ?
I__inference_conv_block_7_layer_call_and_return_conditional_losses_3835658uCDEFG<?9
2?/
)?&
inputs?????????  ?
p
? ".?+
$?!
0??????????
? ?
.__inference_conv_block_7_layer_call_fn_3833774iCDEFG=?:
3?0
*?'
input_1?????????  ?
p 
? "!????????????
.__inference_conv_block_7_layer_call_fn_3833834iCDEFG=?:
3?0
*?'
input_1?????????  ?
p
? "!????????????
.__inference_conv_block_7_layer_call_fn_3835599hCDEFG<?9
2?/
)?&
inputs?????????  ?
p 
? "!????????????
.__inference_conv_block_7_layer_call_fn_3835614hCDEFG<?9
2?/
)?&
inputs?????????  ?
p
? "!????????????
I__inference_conv_block_8_layer_call_and_return_conditional_losses_3834193vHIJKL=?:
3?0
*?'
input_1??????????
p 
? ".?+
$?!
0??????????
? ?
I__inference_conv_block_8_layer_call_and_return_conditional_losses_3834208vHIJKL=?:
3?0
*?'
input_1??????????
p
? ".?+
$?!
0??????????
? ?
I__inference_conv_block_8_layer_call_and_return_conditional_losses_3835710uHIJKL<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
I__inference_conv_block_8_layer_call_and_return_conditional_losses_3835732uHIJKL<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
.__inference_conv_block_8_layer_call_fn_3834118iHIJKL=?:
3?0
*?'
input_1??????????
p 
? "!????????????
.__inference_conv_block_8_layer_call_fn_3834178iHIJKL=?:
3?0
*?'
input_1??????????
p
? "!????????????
.__inference_conv_block_8_layer_call_fn_3835673hHIJKL<?9
2?/
)?&
inputs??????????
p 
? "!????????????
.__inference_conv_block_8_layer_call_fn_3835688hHIJKL<?9
2?/
)?&
inputs??????????
p
? "!????????????
I__inference_conv_block_9_layer_call_and_return_conditional_losses_3834537uMNOPQ=?:
3?0
*?'
input_1??????????
p 
? "-?*
#? 
0?????????d
? ?
I__inference_conv_block_9_layer_call_and_return_conditional_losses_3834552uMNOPQ=?:
3?0
*?'
input_1??????????
p
? "-?*
#? 
0?????????d
? ?
I__inference_conv_block_9_layer_call_and_return_conditional_losses_3835784tMNOPQ<?9
2?/
)?&
inputs??????????
p 
? "-?*
#? 
0?????????d
? ?
I__inference_conv_block_9_layer_call_and_return_conditional_losses_3835806tMNOPQ<?9
2?/
)?&
inputs??????????
p
? "-?*
#? 
0?????????d
? ?
.__inference_conv_block_9_layer_call_fn_3834462hMNOPQ=?:
3?0
*?'
input_1??????????
p 
? " ??????????d?
.__inference_conv_block_9_layer_call_fn_3834522hMNOPQ=?:
3?0
*?'
input_1??????????
p
? " ??????????d?
.__inference_conv_block_9_layer_call_fn_3835747gMNOPQ<?9
2?/
)?&
inputs??????????
p 
? " ??????????d?
.__inference_conv_block_9_layer_call_fn_3835762gMNOPQ<?9
2?/
)?&
inputs??????????
p
? " ??????????d?
J__inference_discriminator_layer_call_and_return_conditional_losses_3834996?9:;<=>?@ABCDEFGHIJKLMNOPQ2>?;
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
J__inference_discriminator_layer_call_and_return_conditional_losses_3835059?9:;<=>?@ABCDEFGHIJKLMNOPQ2>?;
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
J__inference_discriminator_layer_call_and_return_conditional_losses_3835276?9:;<=>?@ABCDEFGHIJKLMNOPQ2=?:
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
J__inference_discriminator_layer_call_and_return_conditional_losses_3835375?9:;<=>?@ABCDEFGHIJKLMNOPQ2=?:
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
/__inference_discriminator_layer_call_fn_3834685?9:;<=>?@ABCDEFGHIJKLMNOPQ2>?;
4?1
+?(
input_1???????????
p 
? "M?J
#? 
0?????????
#? 
1?????????d?
/__inference_discriminator_layer_call_fn_3834933?9:;<=>?@ABCDEFGHIJKLMNOPQ2>?;
4?1
+?(
input_1???????????
p
? "M?J
#? 
0?????????
#? 
1?????????d?
/__inference_discriminator_layer_call_fn_3835118?9:;<=>?@ABCDEFGHIJKLMNOPQ2=?:
3?0
*?'
inputs???????????
p 
? "M?J
#? 
0?????????
#? 
1?????????d?
/__inference_discriminator_layer_call_fn_3835177?9:;<=>?@ABCDEFGHIJKLMNOPQ2=?:
3?0
*?'
inputs???????????
p
? "M?J
#? 
0?????????
#? 
1?????????d?
J__inference_leaky_re_lu_5_layer_call_and_return_conditional_losses_3836277h7?4
-?*
(?%
inputs?????????@@@
? "-?*
#? 
0?????????@@@
? ?
/__inference_leaky_re_lu_5_layer_call_fn_3836272[7?4
-?*
(?%
inputs?????????@@@
? " ??????????@@@?
J__inference_leaky_re_lu_6_layer_call_and_return_conditional_losses_3836363j8?5
.?+
)?&
inputs?????????  ?
? ".?+
$?!
0?????????  ?
? ?
/__inference_leaky_re_lu_6_layer_call_fn_3836358]8?5
.?+
)?&
inputs?????????  ?
? "!??????????  ??
J__inference_leaky_re_lu_7_layer_call_and_return_conditional_losses_3836449j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
/__inference_leaky_re_lu_7_layer_call_fn_3836444]8?5
.?+
)?&
inputs??????????
? "!????????????
J__inference_leaky_re_lu_8_layer_call_and_return_conditional_losses_3836535j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
/__inference_leaky_re_lu_8_layer_call_fn_3836530]8?5
.?+
)?&
inputs??????????
? "!????????????
J__inference_leaky_re_lu_9_layer_call_and_return_conditional_losses_3836621h7?4
-?*
(?%
inputs?????????d
? "-?*
#? 
0?????????d
? ?
/__inference_leaky_re_lu_9_layer_call_fn_3836616[7?4
-?*
(?%
inputs?????????d
? " ??????????d?
J__inference_sequential_10_layer_call_and_return_conditional_losses_3833381?>?@ABG?D
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
J__inference_sequential_10_layer_call_and_return_conditional_losses_3833398?>?@ABG?D
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
J__inference_sequential_10_layer_call_and_return_conditional_losses_3835947x>?@AB??<
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
J__inference_sequential_10_layer_call_and_return_conditional_losses_3835969x>?@AB??<
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
/__inference_sequential_10_layer_call_fn_3833288s>?@ABG?D
=?:
0?-
conv2d_6_input?????????@@@
p 

 
? "!??????????  ??
/__inference_sequential_10_layer_call_fn_3833364s>?@ABG?D
=?:
0?-
conv2d_6_input?????????@@@
p

 
? "!??????????  ??
/__inference_sequential_10_layer_call_fn_3835910k>?@AB??<
5?2
(?%
inputs?????????@@@
p 

 
? "!??????????  ??
/__inference_sequential_10_layer_call_fn_3835925k>?@AB??<
5?2
(?%
inputs?????????@@@
p

 
? "!??????????  ??
J__inference_sequential_11_layer_call_and_return_conditional_losses_3833725?CDEFGH?E
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
J__inference_sequential_11_layer_call_and_return_conditional_losses_3833742?CDEFGH?E
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
J__inference_sequential_11_layer_call_and_return_conditional_losses_3836021yCDEFG@?=
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
J__inference_sequential_11_layer_call_and_return_conditional_losses_3836043yCDEFG@?=
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
/__inference_sequential_11_layer_call_fn_3833632tCDEFGH?E
>?;
1?.
conv2d_7_input?????????  ?
p 

 
? "!????????????
/__inference_sequential_11_layer_call_fn_3833708tCDEFGH?E
>?;
1?.
conv2d_7_input?????????  ?
p

 
? "!????????????
/__inference_sequential_11_layer_call_fn_3835984lCDEFG@?=
6?3
)?&
inputs?????????  ?
p 

 
? "!????????????
/__inference_sequential_11_layer_call_fn_3835999lCDEFG@?=
6?3
)?&
inputs?????????  ?
p

 
? "!????????????
J__inference_sequential_12_layer_call_and_return_conditional_losses_3834069?HIJKLH?E
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
J__inference_sequential_12_layer_call_and_return_conditional_losses_3834086?HIJKLH?E
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
J__inference_sequential_12_layer_call_and_return_conditional_losses_3836095yHIJKL@?=
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
J__inference_sequential_12_layer_call_and_return_conditional_losses_3836117yHIJKL@?=
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
/__inference_sequential_12_layer_call_fn_3833976tHIJKLH?E
>?;
1?.
conv2d_8_input??????????
p 

 
? "!????????????
/__inference_sequential_12_layer_call_fn_3834052tHIJKLH?E
>?;
1?.
conv2d_8_input??????????
p

 
? "!????????????
/__inference_sequential_12_layer_call_fn_3836058lHIJKL@?=
6?3
)?&
inputs??????????
p 

 
? "!????????????
/__inference_sequential_12_layer_call_fn_3836073lHIJKL@?=
6?3
)?&
inputs??????????
p

 
? "!????????????
J__inference_sequential_13_layer_call_and_return_conditional_losses_3834413?MNOPQH?E
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
J__inference_sequential_13_layer_call_and_return_conditional_losses_3834430?MNOPQH?E
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
J__inference_sequential_13_layer_call_and_return_conditional_losses_3836169xMNOPQ@?=
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
J__inference_sequential_13_layer_call_and_return_conditional_losses_3836191xMNOPQ@?=
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
/__inference_sequential_13_layer_call_fn_3834320sMNOPQH?E
>?;
1?.
conv2d_9_input??????????
p 

 
? " ??????????d?
/__inference_sequential_13_layer_call_fn_3834396sMNOPQH?E
>?;
1?.
conv2d_9_input??????????
p

 
? " ??????????d?
/__inference_sequential_13_layer_call_fn_3836132kMNOPQ@?=
6?3
)?&
inputs??????????
p 

 
? " ??????????d?
/__inference_sequential_13_layer_call_fn_3836147kMNOPQ@?=
6?3
)?&
inputs??????????
p

 
? " ??????????d?
I__inference_sequential_9_layer_call_and_return_conditional_losses_3833037?9:;<=I?F
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
I__inference_sequential_9_layer_call_and_return_conditional_losses_3833054?9:;<=I?F
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
I__inference_sequential_9_layer_call_and_return_conditional_losses_3835873y9:;<=A?>
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
I__inference_sequential_9_layer_call_and_return_conditional_losses_3835895y9:;<=A?>
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
.__inference_sequential_9_layer_call_fn_3832944t9:;<=I?F
??<
2?/
conv2d_5_input???????????
p 

 
? " ??????????@@@?
.__inference_sequential_9_layer_call_fn_3833020t9:;<=I?F
??<
2?/
conv2d_5_input???????????
p

 
? " ??????????@@@?
.__inference_sequential_9_layer_call_fn_3835836l9:;<=A?>
7?4
*?'
inputs???????????
p 

 
? " ??????????@@@?
.__inference_sequential_9_layer_call_fn_3835851l9:;<=A?>
7?4
*?'
inputs???????????
p

 
? " ??????????@@@?
%__inference_signature_wrapper_3835436?9:;<=>?@ABCDEFGHIJKLMNOPQ2E?B
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