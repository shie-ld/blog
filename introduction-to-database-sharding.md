
# An Introduction to Database Sharding

Scaling your database as your app grows


With ever-increasing data volumes that an application has to deal with, it is not possible to keep all the data at a single node, as a single server might not be able to handle such a large throughput. The partitioning of the database into multiple nodes is one technique that is effective in this scenario.

Today, we are going to talk about database sharding (partitioning). Sharding has received a lot of attention in recent years. But, many are unfamiliar with this concept.

In this blog, we will be talking about basic concepts, why is it required, and ways to implement it.

## What Is Sharding?

Sharding is partitioning the data from a single data source to multiple partitions in which the structure of each partition is identical to others. Individual partition is also referred to as a shard. Shards can be placed in the same server or different servers.

To understand this better, look at the image below. We have a *users* table that contains users’ info like *name*, *age*, and *country*. We sharded that into two tables, *users_001* and *users_002. *(On what basis we can a shard a table will be discussed in subsequent paragraphs.)

One important thing to note is that the shards *don’t share any data*. So, replication needs to work along sharding to prevent data loss in case a shard goes down.

![Image 2: Sharding of a table](https://cdn-images-1.medium.com/max/2000/1*hAePdvA7UdzIMQjC8LxxhA.png)*Image 2: Sharding of a table*

## Why Do We Need Sharding?

Let us understand this with an example. Suppose, in image 1, instead of three lanes there was only one lane. What would have happened? Traffic would have been moving slowly, there might be congestion.

Similarly, in the database as well, if we read everything from one table, response time might increase with an increase in load and the database can also go down.

Sharding splits the traffic from a single table to multiple tables just like lanes in the image 1.

## How Is Sharding Done?

There are a few sharding techniques, but the popular ones are:

1. Range-based partitioning.

1. Hash partitioning.

### **Range-based partitioning**

In this case, data is sharded (partitioned) based on the range of a key. The data within the same range falls in the same partition.

Choice of the key is very important here because if the choice is not good then data will be distributed unevenly, i.e. one shard might end up containing most of the data.

In image 3, the *Employee *table partition is based on the *age* column. The ranges of age are (0-30), (30-40), and (>40). So, there will be three shards and employee data will be partitioned based on their ages.

![**Image 3: **Range-based partitioning](https://cdn-images-1.medium.com/max/2000/1*_v3cboFtFLcqPLTvMk3_pw.png)***Image 3: **Range-based partitioning*

### **Hash partitioning**

This is also known as key-based partitioning**. **In this case, we pick up a key and pass to a hash function and get the partition, i.e. the hash function can be considered a map from key to partition.

While deciding the hash function, one has to make sure that the data gets uniformly distributed across shards.

For example, in our partition schema in image 2, our key is the ID column. Our hash function picks the last digit of the ID and puts data in the partition accordingly, i.e. ending with 1 goes to *users_001*, ending with 2 goes to *users_002*, and so on.

![**Image 4: **Hash partitioning](https://cdn-images-1.medium.com/max/2000/1*BXnYY2Z-57mmPwcA_XqHdw.png)***Image 4: **Hash partitioning*

## How Is Sharding Implemented?

From an implementation standpoint, there are two ways of doing this:

1. Client-side partitioning.

1. Proxy-assisted partitioning.

### **Client-side partitioning**

The clients know how data is partitioned and directly select the partitions for reading and writing the data.

The advantage of this method is that there is no middle layer. But the disadvantage is that it is not easy to change the number of partitions after it is implemented as all the clients’ code need to be changed.

![**Image 5: **Client-side partitioning](https://cdn-images-1.medium.com/max/2000/1*fAqCMHz2w9H43EK4eV8YBQ.png)***Image 5: **Client-side partitioning*

### **Proxy-assisted partitioning**

In proxy-assisted partitioning, instead of making a direct call to a shard, clients make a request to a proxy server. The proxy server forwards this request to the right shard according to the schema of sharding.

The advantage of this technique is that the client doesn’t know any logic about sharding and the number of shards and partitions can be changed easier than in client-side partitioning.

![**Image 6: **Proxy-assisted partitioning](https://cdn-images-1.medium.com/max/2000/1*THXwyowlceD0PDRWFj6SIg.png)***Image 6: **Proxy-assisted partitioning*

## Conclusion

Sharding can be a great solution for a database with a large amount of data. It helps to split the load from a single node to multiple nodes. But, it adds a lot of complexity to the application.

Sharding can be necessary in some cases, but one need to exhaust other options like adding caching or migrating to a larger server before adding sharding as the time to create and the maintenance costs might outweigh the benefits of sharding.
