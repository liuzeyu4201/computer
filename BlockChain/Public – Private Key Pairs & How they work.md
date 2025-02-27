# Public – Private Key Pairs & How they work



Define： 

Bob : P(b),S(b)

Alice: P(a), S(a)

Mike:P(m),S(m)



## 非对称加密

非对称加密由公钥（public key） 与 私钥（private key）组成

公钥用于加密，私钥用于解密

公钥与私钥唯一对应，由同一协议生成。 P(TX)=hdiaoj ,  S(P(TX))= TX.

公钥对所有人公开，私钥保密。

**例子**

![图2](/home/liuzeyu/LifeChoice/2024-01-24/技能/block_chain/imgs/trans.png)

BOB 使用 alice的公钥加密信息，Servers 收到信息有，bob的公钥，alice的公钥，加密后的信息。

alice收到信息后使用私钥解密。

 problem ： alice和bob的公钥是开放的，任何人都可以使用。可能会存在mike，假借bob'的名义发布信息。

## 数字签名

**原理**： bob首先使用自己的私钥“签名”  TX= "I am bob" , S(TX)=iodhao ,  同时其他信息不变。

当服务器收到信息时，需要使用信息中bob提供的公钥对其反确认（当然不是公钥的初始作用）， P(S(TX))=' I am bob'.

此时确认交易正常，发给alice。

![图2](/home/liuzeyu/LifeChoice/2024-01-24/技能/block_chain/imgs/digtial_sign.png)

## Bitcoin 中的加密
![图2](/home/liuzeyu/LifeChoice/2024-01-24/技能/block_chain/imgs/bit_trans.png)

minner中错误，是A's public key

1. 其中的交易信息是不要加密的只需要 B 签名就行。
2. A 不需要任何操作，也不用验证身份。
3. Minner需要 首先使用 B 的公钥判断，是不是B 本人，在判断是不是合法交易。
