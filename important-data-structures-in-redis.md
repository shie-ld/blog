
# Important data structures to understand in redis

redis data structures
> *Redis is an open source (BSD licensed), in-memory data structure store, used as a database, cache and message broker. It supports data structures such as strings, hashes, lists, sets, sorted sets with range queries, bitmaps, hyperloglogs and geospatial indexes with radius queries.*

**Redis** is the world’s **most popular** in-memory data structure server. In order to make good use of it, we need to understand its basic data structures first.

## Strings

The Redis String type is the simplest type of value you can associate with a Redis key. It is also the base type of the complex types, because a List is a list of strings, a Set is a set of strings, and so forth.

* Create: [SET](https://redis.io/commands/set)

* Retrieve: [GET](https://redis.io/commands/get)

* Update: [SET](https://redis.io/commands/set)
   
* Delete: [DEL](https://redis.io/commands/del)

* Others: [INCR](https://redis.io/commands/incr), [DECR](https://redis.io/commands/decr)

The time complexity of all commands above is O(1).

Note that Redis stores everything in its string representation. Even functions like [INCR](http://redis.io/commands/incr) work by first parsing it into INTEGER then performing the operation.

## Lists

From a very general point of view, a List is just a sequence of ordered elements: 10,20,1,2,3 is a list. But the properties of a List implemented using an Array are very different from the properties of a List implemented using a *Linked List*.

Redis Lists are implemented with ***linked lists* **because for a database system it is crucial to be able to add elements to a very long list in a very fast way.

![linked list example](https://cdn-images-1.medium.com/max/2000/0*mmsvg6TEGzUdP-0Z.png)*linked list example*

From the point of time complexity, Redis List supports constant time O(1) insertion and deletion of a single element near the head and tail, even with many millions of already inserted items. However accessing middle elements is very slow if it is a very big list, as it is an O(N) operation.

O(1) operations:

* Create: [LPUSH](https://redis.io/commands/lpush), [RPUSH](https://redis.io/commands/rpush)

* Delete: [LPOP](https://redis.io/commands/lpop), [RPOP](https://redis.io/commands/rpop)

* Others: [LLEN](https://redis.io/commands/llen)

LLEN has constant time complexity, because It is common for data structures to incorporate a “length” variable which is adjusted every time the structure is modified in time complexity at most equal to the complexity of the modifying algorithm. This way, every time you want to know the length, the data structure simply reads the value of that variable.

O(N) operations:

* Retrieve: [LINDEX](https://redis.io/commands/lindex), [LRANGE](https://redis.io/commands/lrange)

* Update: [LSET](https://redis.io/commands/lset)

Note: while [LRANGE](https://redis.io/commands/lrange) is technically an O(N) command, accessing small ranges towards the head or the tail of the list is a constant-time operation.

### Use Case:

Imagine your home page shows the latest photos published in a photo-sharing social network and you want to speed up access.

Every time a user posts a new photo, we *add its ID* into a list with LPUSH. When users visit the home page, we use LRANGE 0 9 in order to get the* latest 10 posted items.*

Another example is a social news feed system explained in the following post.
[**Design a news feed system**
*Discussion for Interview Question*medium.com](https://medium.com/@bansal_ankur/design-a-news-feed-system-6bf42e9f03fb)

## Sets

Sets stores collections of unique, unsorted string elements. Add, remove, and test for the existence of members is very cheap - constant time regardless of the number of elements contained inside the Set.

If you have a collection of items and it is very important to check for the existence (SISMEMBER) or size of the collection (SCARD) in a very fast way. Set is your first choice. Another cool thing about sets is support for peeking (SRANDMEMBER) or popping random elements (SPOP).

O(1) operations:

* Create: [SADD](https://redis.io/commands/sadd)

* Delete: [SREM](https://redis.io/commands/SREM) [SPOP](https://redis.io/commands/SPOP)

* Retrieve: [SISMEMBER](https://redis.io/commands/SISMEMBER)

* Others: [SCARD](https://redis.io/commands/scard), [SDIFF](https://redis.io/commands/sdiff)

O(N) operations:

* Others: [SUNION](https://redis.io/commands/sunion)

* List: [SMEMBERS](https://redis.io/commands/smembers)

O(…) operations:

* Others: [SINT](https://redis.io/commands/sinter)

### Use Case:

Sets are good for expressing relations between objects. For instance, we can use sets in order to implement many-to-many relationships between posts and tags.

![](https://cdn-images-1.medium.com/max/2596/1*MhT1Jn1VN9d_UiFfXvIb9w.png)

Here are two problems that Redis set can answer easily:

* To get all the posts tagged by MySQL

    > tag:MySQL:posts

* To get all the posts tagged by multiple tags like MySQL, Java and Redis, we user

    > SINTER tag:Java:posts tag:MySQL:posts tag:Redis:posts

## Hash

Redis Hashes are maps between string fields and string values, so they are the perfect data type to represent objects (e.g. A User with a number of fields like name, surname, age, and so forth):

O(1) operations:

* Create: [HSET](https://redis.io/commands/hset)

* Delete:

* Retrieve: [HGET](https://redis.io/commands/hget)

* Others: [HLEN](https://redis.io/commands/hlen)

O(N) operations:

* List: [HKEYS](https://redis.io/commands/hkeys)

### Use Case:

We can use Hash map to model a user from a SQL table

![](https://cdn-images-1.medium.com/max/2984/1*2L5ORp-8KMmM5ChTR3h94A.png)

    > HMSET user:139960061 login dr_josiah id 139960061 followers 176
    OK

    > HGETALL user:139960061
    1) "login"
    2) "dr_josiah"
    3) "id"
    4) "139960061"
    5) "followers"
    6) "176"

    > HINCRBY user:139960061 followers 1
    "177"

## Sorted Sets

Redis Sorted Sets are, similarly to Redis Sets, non-repeating collections of Strings. The difference is that every member of a Sorted Set is associated with a score, that is used in order to take the sorted set ordered, from the smallest to the greatest score.

![redis sorted set example](https://cdn-images-1.medium.com/max/3008/1*mDZJgmJM6Z1fY9ThqGleqg.png)*redis sorted set example*
> With sorted sets you can add, remove, or update elements in a very fast way (in a time proportional to the logarithm of the number of elements). Since elements are *taken in order* and not ordered afterwards, you can also get ranges by score or by rank (position) in a very fast way. Accessing the middle of a sorted set is also very fast, so you can use Sorted Sets as a smart list of non repeating elements where you can quickly access everything you need: elements in order, fast existence test, fast access to elements in the middle!

O(1) operations:

* others: [ZCARD](https://redis.io/commands/zcard)

O(log(N)) operations:

* Create: [ZADD](https://redis.io/commands/zadd)

* Delete: [ZREM](https://redis.io/commands/zrem), [ZPOPMAX](https://redis.io/commands/zpopmax)

* Retrieve: [ZRANGE](https://redis.io/commands/zrange), [ZRANK](https://redis.io/commands/zrank)

### Use Case:

Many Q&A platforms like [Stack Overflow](https://redis.io/topics/whos-using-redis) and Quora use Redis Sorted Sets to rank the highest voted answers for each proposed question to ensure the best quality content is listed at the top of the page.

## Comparison

Let’s compare the three data structure in a table.

    +-----------+--------------------+---------------+---------------
    | structure |   allow duplicate  |  is sorted    |   user case  |
    +-----------+--------------------+---------------+---------------
    |   list    |         YES        |     YES       |  news feed   |
    |   set     |         NO         |     NO        |  tags        |
    | sorted set|         NO         |     YES       |  scoreboard  |
    +-----------+--------------------+---------------+--------------+

If you have interests for more use cases, please read the following post.
[**What is Redis and why with Use case?**
*‘Redis’, which stands for Remote Dictionary Server. According to Redis official, Redis is an open-source (BSD…*medium.com](https://medium.com/@juwelariful1/what-is-redis-and-why-with-use-case-1b294b91e373)
