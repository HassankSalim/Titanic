
4
PlaceholderPlaceholder*
dtype0*
shape: 
�
VariableConst*�
value�B�"��)�����D?����H��$�Q�I��>�j?��>���>��=$E?�䘾���>t�$��w���*>�v?�9(>���<�fJ>���>�l^��g?��'>Ɵ��#������375��
?�--�9�_��?O������ܲ`>j��2�=�Z�� Q�>��>~*侹�?���XI>��~�5%���z�� k>h7M��6ؽ��?ܷ��w;?���=)��<t�������[ >ꁜ>�㞽�Wr�')�;r��<08O���H����>n�ѽE~�F�>�F1��
���N>�5
>�2>��M�Z�t>�\�:�`Y�>�?��M>�>�%�>{v�N��=o��>��d�"��G?�D��
(=�y)?#^:��C#�B��>n�G>�d=B&���=)S>���䈲=��L��&�>����^Ȇ�+ZC>��?0c5�7\�n�l��#�>���>����K;V��=��	>����u�:�����M�rf>D\��5	>Q�5?����!b?�S?���?"N?��x���->O�x���;��=j>'v�>�~�=���=�ڬ�RL�>�fŽ��=��=o0#>�</?�C:>��ξ���>��Ⱦ�'�����\�.>�࿾��V>�i�D������Ya��2�����<����0j>cn�����*
dtype0
I
Variable/readIdentityVariable*
T0*
_class
loc:@Variable
�

Variable_1Const*�
value�B�"���t����*�>��>#lk�$
��+
���u�}���#�>}�9�y��װ>�����>��C?��y=�N��ɫͽ�[��`3�>Fa/>��>��NJ�`:.��#?>��<�ս��7��>�����e=.��>��J=$�þ��>Lj;�s��2�X��_u?V?M4��\Z�[8�>�jؽW�*�վ?R��?�e>+��>�;���"*�>'\�>?l���	���ݼm>�u�>��;?v����5��'�>��`�=�H˻���z����?��P>,5
?]�(�%���c�	?.�;?i�����>�K>�z/��O���C �7�R�nx�=J?=�Dq=��.~C�փ�>�H���>牊=�_�>o���aQ?#�?���>~5�>9-m?��J?}�=�x
��`n>*
dtype0
O
Variable_1/readIdentity
Variable_1*
T0*
_class
loc:@Variable_1
s

Variable_2Const*Q
valueHBF"8�ʴ>�����=e�?>0
?�r$?�&/?E7?�ǃ��̟�`��?~ş�r1?q���*
dtype0
O
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
T0
s

Variable_3Const*
dtype0*Q
valueHBF"<.��W���)!?^+����(]
?�?�H�����<��ϾX>>Z�v>�i��*Ծ��>
O
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
T0
S

Variable_4Const*
dtype0*1
value(B&"wr>���>1��>��>!'>�+ ���A;
O
Variable_4/readIdentity
Variable_4*
T0*
_class
loc:@Variable_4
?

Variable_5Const*
valueB"�O9��´*
dtype0
O
Variable_5/readIdentity
Variable_5*
_class
loc:@Variable_5*
T0
[
MatMulMatMulPlaceholderVariable/read*
transpose_b( *
transpose_a( *
T0
,
AddAddMatMulVariable_3/read*
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
Add_1AddMatMul_1Variable_4/read*
T0

Relu_1ReluAdd_1*
T0
Z
MatMul_2MatMulRelu_1Variable_2/read*
transpose_b( *
T0*
transpose_a( 
0
Add_2AddMatMul_2Variable_5/read*
T0

Relu_2ReluAdd_2*
T0
@
final_output/dimensionConst*
value	B :*
dtype0
K
final_outputArgMaxRelu_2final_output/dimension*
T0*

Tidx0