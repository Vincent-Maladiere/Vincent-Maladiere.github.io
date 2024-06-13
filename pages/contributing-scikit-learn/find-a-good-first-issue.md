# Find a good first issue

Now that you are all set, let’s find good first issues to start contributing!

### 1. Docstring numpydoc validation

[https://github.com/scikit-learn/scikit-learn/issues/21350](https://github.com/scikit-learn/scikit-learn/issues/21350)

This issue is one of the easiest to get started, because it doesn’t require you to deep dive in the code. You will fix docstrings that don’t comply with the numpydoc format.

Choose unformatted functions from the list (also double check comments on the conversation to make sure no-one is already tackling the function you have chosen) and comment on which one you have chosen.

### 2. Estimator _validate_params

[https://github.com/scikit-learn/scikit-learn/issues/23462](https://github.com/scikit-learn/scikit-learn/issues/23462)

I recommend it as a second issue, because it is a gentle start into coding. You will extend parameter validation for estimators.

### 3. Stalled pull request

[Pull requests · scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn/pulls?q=is%3Apr+is%3Aopen+sort%3Aupdated-desc+label%3AStalled)

Once you have done some PR on the 2 issues above, you can try to tackle more involved pull request by looking for the “stalled” label on pull requests. It is basically PRs that require a bit more work to be merged, and often brings a lot of value.

Other labels that you want to check are:

- good first issue
- help wanted
- easy
- moderate

To continue a stalled issue, you need to:

- Check that it is still stalled:
[scikit-learn.org/stable/developers/contributing.html#stalled-pull-requests](https://scikit-learn.org/stable/developers/contributing.html#stalled-pull-requests)
    
    > *If a contributor comments on an issue to say they are working on it, a pull request is expected within 2 weeks (new contributor) or 4 weeks (contributor or core dev), unless an larger time frame is explicitly given. Beyond that time, another contributor can take the issue and make a pull request for it. We encourage contributors to comment directly on the stalled or unclaimed issue to let community members know that they will be working on it.*
    > 
- Continue from the contributor branch when possible
    
    ```bash
    git pull upstream pull/<PULL_REQUEST_NUMBER>/head:<BRANCH_NAME>
    git checkout <BRANCH_NAME>
    ```
    
- Resolve eventual conflicts with the main branch
    
    ```bash
    git fetch upstream
    git merge upstream/main
    ```
    
- Create a new PR indicating “follow-up” or “supersede” in the PR description. Also mention the original issue with `#<ISSUE_NUMBER>` in the description
- Add original author into the change log entry, if any
