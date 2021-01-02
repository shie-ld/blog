
# How to Choose the Right Database for our Application

Making those database decisions a little easier and a little wiser

![](https://cdn-images-1.medium.com/max/2000/0*O8b6zYRF0d19zkCi)

Working on new projects is always super exciting — we have the freedom to design and architect the things any way we want. But this planning, when not done properly, can cause us a lot of pain in the future.

Choosing your app database is one of the critical decisions you’ll have to make, and with this article, I intend to introduce you to various database options — along with listing some pros and cons to help you make wiser database decisions.

## Key**–**Value

Our database is structured just like a JSON object — every key is unique, and each key points to some value.

![](https://cdn-images-1.medium.com/max/2436/1*QnwChq06WzOEtTT8qw059Q.png)

It keeps the data in the memory, which is very fast but comes with a capacity limit, so you can’t store a huge amount of data. And since there’s no disk involved, everything is blazing fast.

There are no queries or joins involved, so there isn’t much data modelling to worry about. Since there’s no schema, the developers always have the flexibility to change data as they may like.

![](https://cdn-images-1.medium.com/max/4828/1*1Czu-WPw2uN2YH6O36sCvQ.png)

### **When to use this technique**

* This technique is mostly used as a caching mechanism for when some part of the data is being very frequently fetched and observed

* Hence, the key–value technique is widely used along with other databases as a caching mechanism

![](https://cdn-images-1.medium.com/max/3372/1*nRatEMZaVXriYa6f2jzHLQ.png)

## Wide Column

This is like key–value but on steroids. The value is modified to store a set of columns instead of just plain data.

![](https://cdn-images-1.medium.com/max/2616/1*3kbSc1usFyQ6iorZog7Kzg.png)

With the introduction of columns, you can now group your related data, but still, there’s no standard schema in place. Hence, each of the keys can point to differently grouped data.

Since there’s no schema, it can handle unstructured data and comes with a query language called CQL, which is similar to SQL but way less powerful.

Data comes in a continuous stream, like from an IoT device, stock market, financial transactions, or your Netflix watch history.

![](https://cdn-images-1.medium.com/max/3024/1*fwlUxozzUNidapixyqhd1Q.png)

### **When to use this technique**

* Frequent writes

* Infrequent updates or reads

This is still not general-purpose. Hence, it can be used for storing historical data from all of our different apps.

## Document-Style Database

This is one of the most popular database techniques we use. This obviously consists of documents, and each document is a set of key**–**value pairs. They’re unstructured and don’t require a schema.

![](https://cdn-images-1.medium.com/max/2572/1*gGzYrc-tVYZvbLRQd_iqsg.png)

Documents are grouped together into *collections, *and these collections can be structured into a logical hierarchy.

This logical collection of data lets you group your related data in a much more logical manner, which seems similar to a relational database.

![](https://cdn-images-1.medium.com/max/3372/1*IrWHcAyqZ4mP4qPW6zBW4A.png)

### **Since our database can’t run join queries, what do we do to get all that related data at once?**

We store it all together! We encourage the denormalization of your database and data duplication/inconsistency is a trade-off we’re ready to make.

Reads are really fast, but writing and updating the data while making sure its consistent can be a bit of a challenge.

The document database is a really great fit for general-purpose apps and might be a good fit for most of the apps, games, and IoT.

If you really aren’t sure about your database schema, then a document database is the best way to start.

### **Popular document-type databases**

![](https://cdn-images-1.medium.com/max/2540/1*9WgYjujH0VueVqqkCWhoMA.png)

The document-style database falls short when you have a lot of data, which might be directly or indirectly related to each other.

For those scenarios, you’ll have to run multiple complex queries and then merge all the received data in the front-end application, or you can use a relational database where these complex queries are managed by the database.

## Relational Database

We all have heard about these databases, and the most popular ones are MySQL, Postgres, and SQL Server. They’ve been here for a while and still are a popular choice for many applications.

We use Structured Query Language (SQL).

### Significance of “relational”

Imagine a car factory where there are different hubs that manufacture the car parts.

Let’s say the doors are manufactured at one place, while the wheels, the body, and the interiors are all made at their own different locations.

![Imaginary car-factory blueprint](https://cdn-images-1.medium.com/max/2660/1*9LV5gItIIC_GPE7rYUAKfA.png)*Imaginary car-factory blueprint*

Each manufactured part has a unique ID assigned to it.

So once a car has to be assembled, you can fetch all of the parts from all of these different locations and assemble your car.

![](https://cdn-images-1.medium.com/max/2952/1*LBmG7-aK_g2c5u_xd-Vgmg.png)

For building such a factory, we’d create the blueprint for such a factory, which makes sure that the overall process of manufacturing the car is very efficient and optimal. This blueprint is called a *schema* when it’s used in a database.

Hence, we need to plan out the schema for our database to make sure our database is also very efficient for our application’s data needs.

### The downside

* Just like how, over time, changing the layout of a car factory to coincide with changing requirements will cost a car company a ton of time and money, it’s a similar situation when a large-scale application has to do so. Make sure to use a relational database when your requirements are clear.

* Also, once you build a plant with the capacity of manufacturing 30 cars a month, you can’t easily scale your plant to make 90 cars a month. Similarly, our relational databases can be harder to scale, but there are some exceptions like Cockroach DB and PostgreSQL** **that are designed to work at scale.

### The good side

* The SQL databases are ACID-compliant, which means our data validity and integrity are never compromised, even though the read-write operation might fail in between — which makes it a great fit for banking/financial-related data

* Once you have a schema in place, you can be assured the data stored will always be stored in a fixed structure following a set of validations, which you’d have defined in the schema

### What’s the best fit for you?

* If your requirements are clear and you’re sure that you won’t be needing any drastic changes to the requirements, go for this one

* If you’re not so sure about the requirements and are in sort of an experimental phase, it’s better to go with a NoSQL database

But what if we don’t need to create a schema and can directly store the relationship as data?

## Graph Database

Here our data is stored in the nodes, and the relationship is defined as edges. It’s pretty sweet! Let’s see how.

If you have to find out all of the students studying computer science in a SQL database, you’ll need a lookup/middleman table that stores records of all the students studying computer science separately.

![](https://cdn-images-1.medium.com/max/2116/1*xdFCeTxVLCWpvjv4iTii8w.png)

In graphs, it’d just be way more straightforward, as we won’t have to store the relationship part of our data separately, and it comes instinctually with this new style.

![Relationships are easier to record and maintain in graphs](https://cdn-images-1.medium.com/max/2000/1*Lhqos0kLLidbb1hW_eqmkA.png)*Relationships are easier to record and maintain in graphs*

With this new way of directly showing the relationship between two nodes, our complex join queries here become much more simple, greatly improving the performance of the database when compared to SQL.

Hence, use such a database when you’re dependent on a lot of join operations and the performance is degrading because of that.

## Searchable Database

If you’re building an application like Google, where on a small-string query search you have to return all of the matching records fast — you’re talking about a full-text search engine.

These databases are based on the Apache Lucene project started in 1999.

Algolia and MeiliSearch are full-text search engines.

They look similar to a document-type database. We have an index, and we add data objects to it. The search-database engine will analyze all of the text in the document and create something called *inverted indexes.*

When you query something, the database only goes and checks in the inverted indexes, which makes the overall process look blazing fast, even for a large database.

I saved the most exciting one for last.

## Multi-Model Database

There are multiple options out there, but the most popular one seems to be [Fauna](https://fauna.com).

As application devs, we usually just care about the JSON, which we can consume in the front end of our apps.

With Fauna, we don’t have to worry about data modelling, the schema, scaling, replication, or normalization and simply just get our JSON data. We define how we want to access our data using GraphQL.

Let’s take an example of a scenario of an Instagram-like app. We’ll define our rules using JSON for a User, Post, and Query.

![](https://cdn-images-1.medium.com/max/3444/1*HrZlm11V4ckfa5zEZkR8rQ.png)

We just uploaded our GraphQL schema — it automatically creates a collection to store data and indexes to query the data.

Behind the scenes, it’s figuring out how to take advantage of different paradigms like relational, graphs, and document based on the GraphQL schema you provided.

We just add our data simply in the same way we did in a document database, and we’re not stuck with the limitations of data modelling.

The best part — this is ACID-compliant and is extremely fast.

You don’t need to worry about the infrastructure. Just define how you need the data, and the cloud will take care of the rest for you.

### Downside

Obviously the pricing is a downside. Great things don’t come free, but they do have generous plans/open-sourced options for devs who want to learn — and small startups as well.

![](https://cdn-images-1.medium.com/max/2668/1*BRN1K9TGOD-k7oOaZh9OqA.png)

Here are some great features worth noting listed by Fauna:

![](https://cdn-images-1.medium.com/max/3844/1*3LWuLpf0dtqW7zH2crNWtw.png)

We’re not done! There’s so much more to learn different databases, but I hope this was a good introduction to the various options we have as developers to use in our apps.

## More Reads About Databases
[**Types of databases**
*Depending upon the usage requirements, there are following types of databases available in the market − Let us explain…*www.tutorialspoint.com](https://www.tutorialspoint.com/Types-of-databases)
[**Full-Text Search in your Database: Algolia vs Elasticsearch | Algolia Blog**
*Most applications and websites are built on top of a database. It can be a traditional relational database like MySQL…*www.algolia.com](https://www.algolia.com/blog/full-text-search-in-your-database-algolia-versus-elasticsearch/)
[**Implementing the Netflix Media Database**
*In the previous blog posts in this series, we introduced the Netflix Media DataBase (NMDB) and its salient “Media…*netflixtechblog.com](https://netflixtechblog.com/implementing-the-netflix-media-database-53b5a840b42a)
[**Should You Use AWS or Firebase for the Back End of Your Mobile App?**
*Which is the right solution for your requirements?*medium.com](https://medium.com/better-programming/should-you-use-aws-or-firebase-for-the-back-end-of-your-mobile-app-a28f93540520)

Thanks for reading. Have a good one!
