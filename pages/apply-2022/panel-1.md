# 28. Panel: Alexander Ratner & Aparna Dhinakaran & Ketan Umare & Biswaroop Palit

[https://www.youtube.com/watch?v=m-qcjCE7AHU](https://www.youtube.com/watch?v=m-qcjCE7AHU)

- What defines success for an ML team?
    
    ⇒ We live in an unstructured and unlabelled data world. The capacity of the team to handle it. A baseline, a metric and a clear understanding aligned with the business team is the first thing to look for. First-mile thing: what is the status of getting the data, and shaping it well?
    
    These fundamentals sound boring but it’s what matters.
    
    ⇒ The ML team velocity. How many shots and goals can this team can have to get ML system in prod. Is it 10 months of research? Shorter?
    
    How do you measure performances and quickly iterate? Successful product and eng teams try to get MVP quickly and have tools in place to iterate.
    
- What about the ML flywheel
    
    ⇒ Tech is never good the first time, but you got to have something out and set up observability, with a clear line of ownership. Lots of parallels to the software world.
    

- Which principles do successful ML teams adopt? Patterns?
    
    ⇒ Focusing on data quality in terms of reliability, testing the data, and observability, are fundamental. ML team need to care about those. Pipeline complexity has increased, data transformation and processes have also increased. Automation around the data pipelines and ML model themselves
    
    We observe that stakeholder buy-in helps a lot with a successful ML team. Understand business KPIs in depth
    
    ⇒ Always start with the simplest things. 
    
    Ownership across the entire team: when DS ship a notebook and ML engineers need to rewrite it. Folks need to work together. Tricky to have a single owner.
    
    ⇒ Uber vs Google approach: totally separated DS vs MLE teams and full single ownership
    
    Both teams need to own the success of the project.
    

- Which successful ML pattern set team from the rest
    
    ⇒ You can’t own everything. Collaboration needs to be the focus. Also, what is the process between these teams? Good relationships? How often do people meet, communicate and measure success?
    
    You shouldn’t rely on some out-of-the-loop expert knowledge for any aspects of data modelling or feature extraction. Instead, have this talent in your team day-to-day.
    
    Being very disciplined to not fall for the last shiny model, focus instead on data quality, causality and observability.
    
    When you start a mature software eng project, you think about maintainability, governance etc but data eng teams, surprisingly, don’t often take this approach. Avoid rushing to the solution before sketching some basic routines and long term objectives.
    
    Have a leaderboard, a metrics or it will fail. But team that are too reductive and glued to that number without having a product/business vision will also fail. You need both.
    
- What type of structure successful ML team have?
    
    ⇒ A lot of individual ML teams, owning different features, and then there are also central ML teams, like platform. Central ML teams try to really understand their customers and their business stakes. Define who owns the infrastructure.
    
    ⇒ ML platform is becoming the norm, we need to push them to be product team, and multidisciplinary. We need to mix MLE and DS with product teams.
    
    ⇒ Teams need to be like application services, being able to communicate easily, like the RPC protocol, a simplified interface for everyone to interact (during human communication but also on implementing data project)
    
- How do you ensure that your customers are really successful (if you’re a SaaS & data company)
    
    ⇒ Successful teams tend to treat data as products themselves. A lot of implications in the cultural aspect of collaboration between teams. Data producer vs data consumer, a lot of clarity is driven regarding ownerships.
    
    We wrote a guide to share best practices. How to configure alerts and notifications to avoid alert fatigue, data quality best practices. We run every consumer through this guide, to arrive at better data quality.