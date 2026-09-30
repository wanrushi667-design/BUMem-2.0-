1. Avoid overengineering, excessively defensive programming, and production-grade hardening. In particular, do not add excessive validation, fallback behavior, compatibility layers, or defensive checks for inputs controlled by this project.
	
2. Avoid speculative generality; that is, do not add unnecessary generality for needs that might or might not arise in the future.
	
3. Avoid premature abstraction. Typically, do not add unnecessarily complicated wrappers unless the functionality needs to be used repeatedly and requires a large chunk of code to implement.