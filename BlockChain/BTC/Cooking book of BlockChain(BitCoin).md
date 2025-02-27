# BlockChain(BitCoin) CookEssay

## quotation                    

<center>Email: 2371912417@qq.com</center>

本文作为加密货币的“简要”，用于**快速** 了解到比特币的逻辑，而不是加密货币的学术文章，所以肯定很多技术问题有待商榷。不过不是本文目标，如果读者在读完本文就可以清楚比特币交易信息中的“名词”，并能够上手交易就达到目的了。


## 1. Hash Function
Hash函数可谓是区块链的关键是区块链安全性的保证。

1.1 def:  A hash function is any function that can be used to map data of arbitrary size to fixed-size values , though there are some hash functions that support variable length output. 
1.1 prop: Collision resistance :即没有通用的方法构造出hash collision。
	对于Hash函数(H),无法找到通用的方法使得 构造出 H(X)=H(Y),if X $\neq$ Y

1.2 prop: Hiding : 无法从值域空间观察到映射空间。
	无法从hash值中获得输入空间中的信息

1.3 puzzle friendly : 事先无法知道Hash function 的分布范围。

1.4 difficult to solve, but easy to verify.

​	挖矿过程困难，其他人验证容易

MD5 码曾经作为Hash加密函数（数模比赛还在使用！），不过现在已经找到通用的方法制造hash collision.

ex1.

```
k = 12345
M = 95
h(12345) = 12345 mod 95 
               = 90

k = 1276
M = 11
h(1276) = 1276 mod 11 
             = 0
```

ex2. 将文字转化为tokens，再embeding到vector space，最后通过hash加密。

## 2. Block Chain

### Block

![图1](/home/liuzeyu/LifeChoice/2024-01-24/技能/block_chain/imgs/chain.png)

区块（blocks）中含有两个部分Block header 和 Block body.

Block header中含有包括：

1. Timestamp： The approximate creation time of the block.
2. Version：A version number to track software/protocol upgrades.
3. Merkle Root：Merkle tree 中根部hash值。
4. Target：The proof of work algorithm difficulty target for this block
5. Nonce（32 bits long）：A counter used for the proof of work algorithm. 用于proof-to-work，后文会提到。
6. Previous Hash Value： 上一个区块的hash值。

Block body中包含：transaction list（交易信息）。

### Data strcution

Hash Pointers: 
It comprised of two parts:
		1. Pointer to where some information is stored.
		2. Cryptographic hash of that information.

A hash pointer is a pointer to where data is stored, accompanied by the cryptographic hash of the data, allowing for data verification


(Ps:这里有问题,需要明白什么是指针，不会数据结构的锅。)

​	Block chain is a linked list using hash pointers.

​		因为hash指针中Crytographic hash的作用改变区块中任何信息都会引起most recent Block中hash value的变化。因此每个全节点只需保存一块链即可，需要时可以向前节点请求。

Merkle Tree：

Block Body中的数据结构用于保存交易数据。TX：Transaction exchange，即具体的交易数据。被保存在Body中。

![图2](/home/liuzeyu/LifeChoice/2024-01-24/技能/block_chain/imgs/MerkleTree.png)



比特币中有节点的概念，轻节点（light nodes）类似于手机节点，只保存少量信息，需要配合全节点（full nodes）完成操作。

当轻节点想要确保一个交易已经写入区块链中需要向全节点请求蓝色hash value。向上计算root hash value 并于 header中的Root hash值比较,即可确认是否完成交易。
![图2](/home/liuzeyu/LifeChoice/2024-01-24/技能/block_chain/imgs/Tx.png)



## 3. Decentration

下面使用区块链构建加密货币系统：需要回答两个问题：1. 谁发行货币   2. 如何验证交易有校性？


非对称加密体系： Public Key, Private Key. 公钥用于加密，私钥用于签名。

![图3](/home/liuzeyu/LifeChoice/2024-01-24/技能/block_chain/imgs/key.png)

假设A 发布了一个向B转入3个比特币的交易，交易中A需要广播A的公钥，B的公钥类似于银行账户。

并且需要自己的签名即通过自己的私钥对交易加密并广播，其他节点收到后使用信息后通过A的公钥解密读出交易信息。这时就可以验证是A本人的发布，并验证是否有效，如有效则获得记账权的矿工发布到区块链中。



有效交易：

发行加密货币要解决两个问题 1.double spending attack，2. “无中生有”。

![图4](/home/liuzeyu/LifeChoice/2024-01-24/技能/block_chain/imgs/exchange.png)

在比特币交易中存在两种Hash Pointers，第一个在上文已经提到用于链接区块。在货币交易中引出第二个hash指针，用于记录货币来源。  在交易中供币方需要提供双方的公钥以及自己的货币来源，并用自己的私钥进行签名。

比如交易中区块二，A->B(5)这个交易中，A必须说明自己的货币来自于第一块的铸币奖励。 B->D(4)这个操作，B也必须说明自己的Coin来自于上一个操作。这些过程均有Poiners存储信息，并由Hash函数保证不可“篡改"（这里的安全是指逻辑上的，下文在分布式共识中会解释）。 现在在蓝色部分B已经无法说明自己的货币如何来的因此交易不会被“诚实”节点写入。

但是当区块中记录交易复杂或者某个节点在交易一次后沉默，并在多年后重新交易，因为区块的延长将难易判断，因此Bitcoin中引入了：

UTXO ：Unspent Transaction Output（set），集合将记录整个区块链中未被交易的全部信息。 这里的信息包括：

1. 产生输出交易时的HASH值。
2. 目标在交易中是第几个输出。

例子：

![图5](/home/liuzeyu/LifeChoice/2024-01-24/技能/block_chain/imgs/UTXO.png)



可以看出在上图交易中B D 的信息被保存在UTXO集合中，当下次交易时其他区块只需要查询UTXO集合中信息即可，而不用去区块链中寻找。UTXO集合是别全节点维护这也是全节点的职责。



###  Distributed consensus

根据比特币设计 首先某个节点广播出交易信息，当某个全节点收到信息后计算和验证交易有效性，并将交易写在自己的“候选“区块账本中，同时全节点运行“挖矿操作”，当解决puzzle后发布候选区块。当其他节点听到有人发布区块后就会停止手上工作，验证这个区块合法性。如果不合法，诚实节点将会不理睬继续挖矿；如果合法将会结束上区块的挖矿遵循新区块开挖。

解释： 1. 矿工遵循延长最长合法链原则。2. 题目包含上一个Previous Hash Value 值，因此必须重新开始挖矿。3. 比特币问题存在progress free性质，是否解出问题与已经消耗的时间无关。因此如果没有解出上一个问题也不会可惜。



在分布式系统中没有人有绝对权力，因此判断一个交易是否承认需要被设计，传统的多数投票法可能会被 Sybil attack 。在比特币系统中被设计为计算力投票的方式，有计算能力的节点通过尝试Block Header中Nonce值证明自己的计算力，当解决了问题时可以发布自己的账本也被称为获得了“记账权”。 具体形式是寻找 block header中的 nonce值 使得满足 Hash(block header)  &le; target. 



等待上链集合： 显然根据不同的节点收到交易的先后不同，因此储存在自己账本的时间也不同。因此会存在等待上链集合，用于“缓存”。当某节点发现自己的缓存中的某个记录被的获得记账权节点已经发布到区块中就会清除，或者如果因为新上链的交易会导致自己缓存中的交易非法，也会清除自己的记录。因为所有诚实节点只会维护最长区块链。

### Mining

比特币系统中只有一种方式发行货币及通过Block Reward。

比特币系统对矿工的奖励为（proof-of-work）。由两部分构成 Block Reward和记账奖励。

初块奖励： 当矿工解出问题获得记账权后将会获得系统的初块奖励，根据系统设计每21万个区块后奖励折半，目前(15-3-2024)初块奖励为6.25个Bitcoin。根据计算系统中比特币总量为21W * 50 * (1+1/2+1/4+1/8.....) =2100W个。当比特币枯竭后将会进入记账奖励阶段。记账奖励：即收取交易费。
![图3](/home/liuzeyu/LifeChoice/2024-01-24/技能/block_chain/imgs/fullnodes.png)





## 实例：


![图6](/home/liuzeyu/LifeChoice/2024-01-24/技能/block_chain/imgs/example.png)


|LeftSide：|RightSide|
|---|---|
|Hash：区块的hash值|From：供币方地址|
|Distance已经上链时间|To：收币方地址|
|BTC：比特币交易量|多收入地址：转入给多个人或找零地址|
|Input Value:  供币量|多输入地址：某一个地址余额不足。|
|output value :Input+Block Reward ||
|Merkle Root :  Merkle tree的根部hash值，header中只需保存根部值即可确保交易数据“不会被修改”||
|Mined : 目前区块奖励||

- **私钥**：私钥是一个包含64位随机数的关键，用于生成公钥和地址。私钥必须被安全地保存，因为拥有私钥就意味着对相应地址中的比特币进行操作的权限。
- **公钥**：公钥是由私钥通过椭圆曲线加密算法生成的，通常是一个65个byte的数组。拥有私钥可以计算出对应的公钥，但反过来却无法从公钥计算出私钥。
- **地址**：在数字货币交易中，地址用于接收转账。地址是由公钥经过数字签名和哈希算法运算得到的。因此，地址不等于公钥，而是公钥的另一种表现形式。









<font size=5>Bitcoin~GOLD:</font>

**比特币在市场支撑了区块链的交易市场，但同时没有产生信息渠道, 而且由于设计的稀缺性会导致对交易次数影响。** 



比特币每个区块4 M 代表他无法储存大量数据。

Hashrate： 24/6/23 : 全球算力609.66 EH/s ,
https://www.coinwarz.com/mining/bitcoin/hashrate-chart

difficulty : 难度度量~ 挖矿竞争度
https://www.coinwarz.com/mining/bitcoin/difficulty-chart
![图2](/home/liuzeyu/LifeChoice/2024-01-24/技能/block_chain/imgs/cal.png)





对个体挖矿者出块时长与 个人算力在全球比例，和问题难度有关，但系统通过算法将难度维持在10min左右。

最先进矿机：Antminer S21 Hyd (335Th) .  功率：5360W，价格6,091刀

<font size=4>Minner:</font>

矿工并不影响coin价格，反而被价格绑定
美国德州商业用电 0.0878 美分/千瓦时

单个矿机 需要35年才能获得初块奖励 （根据算力在总算力的比值）

|machine|price|year|ele|sun|
|---|---|---|---|---|
|35| 257,953 |1|144,288|357,473|
|140| 938014  |3 mouthA|189,749|1,042,489|


|Coin price|count|return cycle 35|return cycle 140|
|---|---|---|---|
|64,430|3.25|3.9 year|10.85|
|70,000|3.25|3 year|6|
|60,000|3.25|5|10+|



|balance point|lowest point|
|---|---|
|60,000|44,396|

![图2](/home/liuzeyu/LifeChoice/2024-01-24/技能/block_chain/imgs/effect.png)

**电价和比特币价格是影响比特币底层存在性的因素（周期）**



**问题点**

1.  区块奖励下降导致交易费上升
2.  比特币价格取决于市场信心，市场信心由谁保障？
3.  缺少包含更多信息






## reference
http://zhenxiao.com/blockchain/
https://www.blockchain.com/explorer/blocks/btc/834739bitcoin price