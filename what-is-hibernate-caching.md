
# What Is Hibernate Caching? Introduction of First-Level & Second-Level Cache

Hello Readers, Greetings for the day.!

In this article, we will take a look at Hibernate caching, and also will get an idea about first-level cache and second-level cache.

## **Let’s start as a noobie, what is caching?**

In general, caching is a mechanism to store copies of data or files in such a manner so that they can be served quickly. In computer science caching is related to hardware or software component that stores data so that future requests for that data can be served faster; the data stored in a cache might be the result of an earlier computation or a copy of data stored elsewhere. — [Wiki](https://en.wikipedia.org/wiki/Cache_(computing))


If we are talking about database cache then caching will act as a buffered memory that remains between the application and the database. It stores recently demanded/inquired data in system memory to reduce the numbers of calls to the actual database.

Now, I hope you are clear with the caching.. so let's take the next step to understand **what is Hibernate caching and its worthiness**?

Hibernate is an ORM (Object-relational model) tool that is widely used by developers worldwide. It has many in-built futures available that make the developer’s life simple. One of them is a ***caching mechanism***.

Hibernate caching acts as a layer between the actual database and your application. It reduces the time taken to obtain the required data — as it fetches from memory instead of directly hitting the database. It is very useful when you need to fetch the same kind of data multiple times.

### There are mainly two types of caching:

* First level cache

* Second-level cache

### **1). First level cache**

The First level cache is ***by default enabled*** by Hibernate itself. The session object maintains the first-level cache.

An application can have many sessions. Data hold by one session object is not accessible to the entire application — means the data of a particular session is not shared with other sessions of the application. So you can use the first-level cache to store local data i.e. required by the session itself.

Let’s make it simple to understand, As we all know that hibernate is an ORM tool that is used to simplify DB operations. It “converts convert objects to relations (to store into DB) and vice-versa”.

So when you query an entity or object, for the very first time it is retrieved from the database and stored in the first-level cache (associated with the hibernate session). If we query for the same entity or object again with the same session object, it will be loaded from cache and no SQL query will be executed. Take a look at the below code snippet.

```sh
// We have one record in DB with the Employee details like, 101, John Doe, UK

// Open hibernate session
Session session = HibernateUtil.getSessionFactory().openSession();
session.beginTransaction();

// Fetch an Employee entity from the database very first time
Employee employee = (Employee) session.load(Employee.class, empId);
System.out.println("First call output : " + employee.getName());
 
// Request for Employee entity again
employee = (Employee) session.load(Employee.class, empId);
System.out.println("Second call output : "employee.getName());
 
session.getTransaction().commit();
HibernateUtil.shutdown();
 
// Output:
// First call output : John Doe
// Second call output : John Doe
```

In the above example hibernate will fire query only a single time to the Database. From the second time onwards it will return only from the session object.

### Some useful methods:

* ***Session.evict():*** to remove the cached/stored entity.

* ***refresh():*** method to refresh the cache.

* ***clear():*** method to remove all the entities from the cache.

**Note:** *You **can not disable** the first-level cache, it is enabled by the hibernate itself. *Hibernate entities or database rows remain in cache only until Session is open, once Session is closed, all associated cached data is lost.
> Keep in mind that caching at the ***session-level (first-level) ***has some memory impacts, especially when you’ve loaded a bunch of a large objects. Long-lived sessions with several large objects consume more memory and may cause out of memory errors.

### 2) Second level cache

The second-level cache is ***by default disabled, ***the developer needs to enable it explicitly, and the SessionFactory object is responsible to maintain it. The second-level cache is accessible by the entire application means data hold by SessionFactory can be accessible to all the sessions. Keep in mind that, once the session factory is closed all the cache associated with that is also removed from the memory.

Let’s take an example: Suppose your application has 2 active sessions session1 and session2 respectively. Now, session1 has requested data having id=101 so that will be fetched from a database as it is the first call, and then it is stored into the second-level (SessionFactory) as well the first-level (session) cache also. Now, session2 requires the same data so it has also been queried with the same id=101. So this time session2 will get data from the SessionFactory, it will not going to hit the database. Take a look at the below code snippet.

```sh
// Open hibernate session
Session session = HibernateUtil.getSessionFactory().openSession();
session.beginTransaction();

// Employee entity is fecthed very first time (It will be cached in both first-level and second-level cache)
Employee employee = (Employee) session.load(Employee.class, empId);
System.out.println(employee.getName());

// Fetch the employee entity again
employee = (Employee) session.load(Employee.class, empId);
System.out.println(employee.getName()); //It will return from the first-level

// Evict from first level cache (That will remove employee object from first-level cache)
session.evict(employee);

// Fetch same entity again using same session
employee = (Employee) session.load(Employee.class, empId);
System.out.println(employee.getName()); //It will return from the second-level

// Fetch same entity again using another session
employee = (Employee) anotherSession.load(Employee.class, empId);
System.out.println(employee.getName());//It will return from the second-level

System.out.println("Response from the first-level : " + HibernateUtil.getSessionFactory().getStatistics().getEntityFetchCount());
System.out.println("Response from the second-level : " + HibernateUtil.getSessionFactory().getStatistics().getSecondLevelCacheHitCount());
 
// Output:
// Response from the first-level : 1
// Response from the second-level : 2

```

As you can find in the above snippet that you got a response from the second-level cache when an object is evicted from the first-level cache.

**How it works (in short):** When hibernate session try to load an entity, it will first find into the first-level cache, if it does not found then it will look into the second-level cache and return the response (if available), but before returning the response it will store that object/data into first-level also so next time no need to come at the session-level. When data is not found in the second-level then it will go to the database to fetch data. Before returning a response to the user it will store that object/data into both levels of cache so next time it will be available at cache stages only.

**Note:** Hibernate does not provide any default implementation for the second-level cache. It gives ***CacheProvider*** interface, so any third party Cache which implements CacheProvider interface can be hooked as Second level cache, like ***EHCache or NCache***.

### One small question, where the cache is stored?

In the Hibernate session is the hibernate’s first-level cache and SessionFactory is a second-level cache. So both (session/session-factory) are objects in a heap area. That means the cache is stored in the RAM only. And because of that, it gives faster access to data rather than databases.

Hope you guys enjoyed reading this, stay tuned.*
