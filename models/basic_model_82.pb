
4
PlaceholderPlaceholder*
dtype0*
shape: 
�
VariableConst*
dtype0*�
value�B�"�-�Q�k�"���-�JE�>1m�>$A�x��=0ǣ>�����@*�=��	�b��ۿU1���n�H�>�Y�> G���?Ce��#��+����Xt����4n��q�?�<��Ӝ���Fy?��?iE��k�@X��8b?
�?B}� p4?#�˿�,�>TN�? �?��?�3�?��=r?�yN?|�@���������`�����s?w��H?
I
Variable/readIdentityVariable*
_class
loc:@Variable*
T0
c

Variable_1Const*A
value8B6"(���?6�$�j#?�ؾ%A�>����R��?>
��R�Z��?*
dtype0
O
Variable_1/readIdentity
Variable_1*
_class
loc:@Variable_1*
T0
K

Variable_2Const*)
value B"�33��6B?ǀC��Ι>���*
dtype0
O
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
T0
?

Variable_3Const*
dtype0*
valueB"��>�F?�
O
Variable_3/readIdentity
Variable_3*
T0*
_class
loc:@Variable_3
[
MatMulMatMulPlaceholderVariable/read*
transpose_b( *
transpose_a( *
T0
,
AddAddMatMulVariable_2/read*
T0

ReluReluAdd*
T0
X
MatMul_1MatMulReluVariable_1/read*
transpose_b( *
transpose_a( *
T0
0
Add_1AddMatMul_1Variable_3/read*
T0

Relu_1ReluAdd_1*
T0
@
final_output/dimensionConst*
value	B :*
dtype0
K
final_outputArgMaxRelu_1final_output/dimension*
T0*

Tidx0