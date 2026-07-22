.PHONY: all engine windhover-engine kestrel-engine test-oracle glm portable test check cuda-test clean install uninstall clean-engine

# Product binary (engine/)
all engine windhover-engine kestrel-engine:
	$(MAKE) -C engine ARCH=$(or $(ARCH),native)

test-oracle:
	$(MAKE) -C engine test-oracle ARCH=$(or $(ARCH),native)

# Legacy colibrì tree (reference only)
glm portable test check cuda-test install uninstall:
	$(MAKE) -C c $@

clean:
	$(MAKE) -C engine clean
	$(MAKE) -C c clean

clean-engine:
	$(MAKE) -C engine clean
