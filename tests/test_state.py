from unittest import mock
from typing import Dict, List, Tuple

import pytest
from plotly.graph_objects import Figure

from pynecone.base import Base
from pynecone.constants import RouteVar
from pynecone.event import Event
from pynecone.state import State
from pynecone.utils import format
from pynecone.var import BaseVar, ComputedVar


class Object(Base):
    """A test object fixture."""

    prop1: int = 42
    prop2: str = "hello"


class TestState(State):
    """A test state."""

    # Set this class as not test one
    __test__ = False

    num1: int
    num2: float = 3.14
    key: str
    array: List[float] = [1, 2, 3.14]
    mapping: Dict[str, List[int]] = {"a": [1, 2, 3], "b": [4, 5, 6]}
    obj: Object = Object()
    complex: Dict[int, Object] = {1: Object(), 2: Object()}
    fig: Figure = Figure()

    @ComputedVar
    def sum(self) -> float:
        """Dynamically sum the numbers.

        Returns:
            The sum of the numbers.
        """
        return self.num1 + self.num2

    @ComputedVar
    def upper(self) -> str:
        """Uppercase the key.

        Returns:
            The uppercased key.
        """
        return self.key.upper()

    def do_something(self):
        """Do something."""
        pass


class ChildState(TestState):
    """A child state fixture."""

    value: str
    count: int = 23

    def change_both(self, value: str, count: int):
        """Change both the value and count.

        Args:
            value: The new value.
            count: The new count.
        """
        self.value = value.upper()
        self.count = count * 2


class ChildState2(TestState):
    """A child state fixture."""

    value: str


class GrandchildState(ChildState):
    """A grandchild state fixture."""

    value2: str

    def do_nothing(self):
        """Do something."""
        pass


@pytest.fixture
def test_state() -> TestState:
    """A state.

    Returns:
        A test state.
    """
    ts = TestState()  # type: ignore
    # For testing purposes, clear these mappings so test cases don't depend on each other
    for state in [ts, *ts.substates.values()]:
        state.substate_var_dependencies.clear()
        state.computed_var_dependencies.clear()
        state.dict()  # re-prime actual relationships
    return ts


@pytest.fixture
def child_state(test_state) -> ChildState:
    """A child state.

    Args:
        test_state: A test state.

    Returns:
        A test child state.
    """
    child_state = test_state.get_substate(["child_state"])
    assert child_state is not None
    return child_state


@pytest.fixture
def child_state2(test_state) -> ChildState2:
    """A second child state.

    Args:
        test_state: A test state.

    Returns:
        A second test child state.
    """
    child_state2 = test_state.get_substate(["child_state2"])
    assert child_state2 is not None
    return child_state2


@pytest.fixture
def grandchild_state(child_state) -> GrandchildState:
    """A state.

    Args:
        child_state: A child state.

    Returns:
        A test state.
    """
    grandchild_state = child_state.get_substate(["grandchild_state"])
    assert grandchild_state is not None
    return grandchild_state


def test_base_class_vars(test_state):
    """Test that the class vars are set correctly.

    Args:
        test_state: A state.
    """
    fields = test_state.get_fields()
    cls = type(test_state)

    for field in fields:
        if field in (
            "parent_state",
            "substates",
            "dirty_vars",
            "dirty_backend_vars",
            "dirty_substates",
            "router_data",
        ):
            continue
        prop = getattr(cls, field)
        assert isinstance(prop, BaseVar)
        assert prop.name == field

    assert cls.num1.type_ == int
    assert cls.num2.type_ == float
    assert cls.key.type_ == str


def test_computed_class_var(test_state):
    """Test that the class computed vars are set correctly.

    Args:
        test_state: A state.
    """
    cls = type(test_state)
    vars = [(prop.name, prop.type_) for prop in cls.computed_vars.values()]
    assert ("sum", float) in vars
    assert ("upper", str) in vars


def test_class_vars(test_state):
    """Test that the class vars are set correctly.

    Args:
        test_state: A state.
    """
    cls = type(test_state)
    assert set(cls.vars.keys()) == {
        "num1",
        "num2",
        "key",
        "array",
        "mapping",
        "obj",
        "complex",
        "sum",
        "upper",
        "fig",
    }


def test_event_handlers(test_state):
    """Test that event handler is set correctly.

    Args:
        test_state: A state.
    """
    expected = {
        "do_something",
        "set_array",
        "set_complex",
        "set_fig",
        "set_key",
        "set_mapping",
        "set_num1",
        "set_num2",
        "set_obj",
    }

    cls = type(test_state)
    assert set(cls.event_handlers.keys()).intersection(expected) == expected


def test_default_value(test_state):
    """Test that the default value of a var is correct.

    Args:
        test_state: A state.
    """
    assert test_state.num1 == 0
    assert test_state.num2 == 3.14
    assert test_state.key == ""
    assert test_state.sum == 3.14
    assert test_state.upper == ""


def test_computed_vars(test_state):
    """Test that the computed var is computed correctly.

    Args:
        test_state: A state.
    """
    test_state.num1 = 1
    test_state.num2 = 4
    assert test_state.sum == 5
    test_state.key = "hello world"
    assert test_state.upper == "HELLO WORLD"


def test_dict(test_state):
    """Test that the dict representation of a state is correct.

    Args:
        test_state: A state.
    """
    substates = {"child_state", "child_state2"}
    assert set(test_state.dict().keys()) == set(test_state.vars.keys()) | substates
    assert (
        set(test_state.dict(include_computed=False).keys())
        == set(test_state.base_vars) | substates
    )


def test_default_setters(test_state):
    """Test that we can set default values.

    Args:
        test_state: A state.
    """
    for prop_name in test_state.base_vars:
        # Each base var should have a default setter.
        assert hasattr(test_state, f"set_{prop_name}")


def test_class_indexing_with_vars():
    """Test that we can index into a state var with another var."""
    prop = TestState.array[TestState.num1]
    assert str(prop) == "{test_state.array.at(test_state.num1)}"

    prop = TestState.mapping["a"][TestState.num1]
    assert str(prop) == '{test_state.mapping["a"].at(test_state.num1)}'


def test_class_attributes():
    """Test that we can get class attributes."""
    prop = TestState.obj.prop1
    assert str(prop) == "{test_state.obj.prop1}"

    prop = TestState.complex[1].prop1
    assert str(prop) == "{test_state.complex[1].prop1}"


def test_get_parent_state():
    """Test getting the parent state."""
    assert TestState.get_parent_state() is None
    assert ChildState.get_parent_state() == TestState
    assert ChildState2.get_parent_state() == TestState
    assert GrandchildState.get_parent_state() == ChildState


def test_get_substates():
    """Test getting the substates."""
    assert TestState.get_substates() == {ChildState, ChildState2}
    assert ChildState.get_substates() == {GrandchildState}
    assert ChildState2.get_substates() == set()
    assert GrandchildState.get_substates() == set()


def test_get_name():
    """Test getting the name of a state."""
    assert TestState.get_name() == "test_state"
    assert ChildState.get_name() == "child_state"
    assert ChildState2.get_name() == "child_state2"
    assert GrandchildState.get_name() == "grandchild_state"


def test_get_full_name():
    """Test getting the full name."""
    assert TestState.get_full_name() == "test_state"
    assert ChildState.get_full_name() == "test_state.child_state"
    assert ChildState2.get_full_name() == "test_state.child_state2"
    assert GrandchildState.get_full_name() == "test_state.child_state.grandchild_state"


def test_get_class_substate():
    """Test getting the substate of a class."""
    assert TestState.get_class_substate(("child_state",)) == ChildState
    assert TestState.get_class_substate(("child_state2",)) == ChildState2
    assert ChildState.get_class_substate(("grandchild_state",)) == GrandchildState
    assert (
        TestState.get_class_substate(("child_state", "grandchild_state"))
        == GrandchildState
    )
    with pytest.raises(ValueError):
        TestState.get_class_substate(("invalid_child",))
    with pytest.raises(ValueError):
        TestState.get_class_substate(
            (
                "child_state",
                "invalid_child",
            )
        )


def test_get_class_var():
    """Test getting the var of a class."""
    assert TestState.get_class_var(("num1",)) == TestState.num1
    assert TestState.get_class_var(("num2",)) == TestState.num2
    assert ChildState.get_class_var(("value",)) == ChildState.value
    assert GrandchildState.get_class_var(("value2",)) == GrandchildState.value2
    assert TestState.get_class_var(("child_state", "value")) == ChildState.value
    assert (
        TestState.get_class_var(("child_state", "grandchild_state", "value2"))
        == GrandchildState.value2
    )
    assert (
        ChildState.get_class_var(("grandchild_state", "value2"))
        == GrandchildState.value2
    )
    with pytest.raises(ValueError):
        TestState.get_class_var(("invalid_var",))
    with pytest.raises(ValueError):
        TestState.get_class_var(
            (
                "child_state",
                "invalid_var",
            )
        )


def test_set_class_var():
    """Test setting the var of a class."""
    with pytest.raises(AttributeError):
        TestState.num3  # type: ignore
    TestState._set_var(BaseVar(name="num3", type_=int).set_state(TestState))
    var = TestState.num3  # type: ignore
    assert var.name == "num3"
    assert var.type_ == int
    assert var.state == TestState.get_full_name()


def test_set_parent_and_substates(test_state, child_state, grandchild_state):
    """Test setting the parent and substates.

    Args:
        test_state: A state.
        child_state: A child state.
        grandchild_state: A grandchild state.
    """
    assert len(test_state.substates) == 2
    assert set(test_state.substates) == {"child_state", "child_state2"}

    assert child_state.parent_state == test_state
    assert len(child_state.substates) == 1
    assert set(child_state.substates) == {"grandchild_state"}

    assert grandchild_state.parent_state == child_state
    assert len(grandchild_state.substates) == 0


def test_get_child_attribute(test_state, child_state, child_state2, grandchild_state):
    """Test getting the attribute of a state.

    Args:
        test_state: A state.
        child_state: A child state.
        child_state2: A child state.
        grandchild_state: A grandchild state.
    """
    assert test_state.num1 == 0
    assert child_state.value == ""
    assert child_state2.value == ""
    assert child_state.count == 23
    assert grandchild_state.value2 == ""
    with pytest.raises(AttributeError):
        test_state.invalid
    with pytest.raises(AttributeError):
        test_state.child_state.invalid
    with pytest.raises(AttributeError):
        test_state.child_state.grandchild_state.invalid


def test_set_child_attribute(test_state, child_state, grandchild_state):
    """Test setting the attribute of a state.

    Args:
        test_state: A state.
        child_state: A child state.
        grandchild_state: A grandchild state.
    """
    test_state.num1 = 10
    assert test_state.num1 == 10
    assert child_state.num1 == 10
    assert grandchild_state.num1 == 10

    grandchild_state.num1 = 5
    assert test_state.num1 == 5
    assert child_state.num1 == 5
    assert grandchild_state.num1 == 5

    child_state.value = "test"
    assert child_state.value == "test"
    assert grandchild_state.value == "test"

    grandchild_state.value = "test2"
    assert child_state.value == "test2"
    assert grandchild_state.value == "test2"

    grandchild_state.value2 = "test3"
    assert grandchild_state.value2 == "test3"


def test_get_substate(test_state, child_state, child_state2, grandchild_state):
    """Test getting the substate of a state.

    Args:
        test_state: A state.
        child_state: A child state.
        child_state2: A child state.
        grandchild_state: A grandchild state.
    """
    assert test_state.get_substate(("child_state",)) == child_state
    assert test_state.get_substate(("child_state2",)) == child_state2
    assert (
        test_state.get_substate(("child_state", "grandchild_state")) == grandchild_state
    )
    assert child_state.get_substate(("grandchild_state",)) == grandchild_state
    with pytest.raises(ValueError):
        test_state.get_substate(("invalid",))
    with pytest.raises(ValueError):
        test_state.get_substate(("child_state", "invalid"))
    with pytest.raises(ValueError):
        test_state.get_substate(("child_state", "grandchild_state", "invalid"))


def test_set_dirty_var(test_state):
    """Test changing state vars marks the value as dirty.

    Args:
        test_state: A state.
    """
    # Initially there should be no dirty vars.
    assert test_state.dirty_vars == set()

    # Setting a var should mark it as dirty.
    test_state.num1 = 1
    assert test_state.dirty_vars == {"num1"}

    # Setting another var should mark it as dirty.
    test_state.num2 = 2
    assert test_state.dirty_vars == {"num1", "num2"}

    # Cleaning the state should remove all dirty vars.
    test_state.clean()
    assert test_state.dirty_vars == set()


class BackendState(State):
    _backend_var: int = 0
    _backend_var_private: int = 0

    @ComputedVar
    def backend_var_rendered(self) -> str:
        """Cast a backend variable to str.

        Returns:
            str representation of _backend_var.
        """
        return str(self._backend_var)

    @ComputedVar
    def backend_var_private_rendered(self) -> str:
        """Cast a private backend variable to str.

        Returns:
            str representation of _backend_var.
        """
        return str(self._backend_var_private)

    @ComputedVar
    def backend_both(self) -> str:
        """Cast both backend variables to str.

        Returns:
            str concatenating _backend_var and _backend_var_private.
        """
        return str(self._backend_var_private) + str(self._backend_var)

    @ComputedVar
    def computed_both(self) -> str:
        """Concatenate both computed variables to str.

        Returns:
            str concatenating rendered computed functions.
        """
        return self.backend_var_private_rendered + self.backend_var_rendered


class BackendChildState(BackendState):
    """Another child state."""

    @ComputedVar
    def backend_var_processed(self) -> int:
        """Calculate some value based on backend var in parent.

        Returns:
            _backend_var times 3
        """
        return self._backend_var * 3


@pytest.fixture
def backend_state_with_substate():
    """A state and substate with only backend and computed vars.

    Returns:
        (state, substate)
    """
    ts = BackendState()  # type: ignore
    ss = ts.get_substate(["backend_child_state"])
    assert ss is not None
    # For testing purposes, clear these mappings so test cases don't depend on each other
    for state in [ts, *ts.substates.values()]:
        state.substate_var_dependencies.clear()
        state.computed_var_dependencies.clear()
        state.dict()  # re-prime actual relationships
    return ts, ss


def test_set_dirty_var_backend(backend_state_with_substate):
    """Test changing state backend vars updates ComputedVar.

    Args:
        backend_state_with_substate: A tuple of (state, substate).
    """
    backend_state, _ = backend_state_with_substate
    # Initially there should be no dirty vars.
    assert backend_state.dirty_vars == set()
    assert backend_state.dirty_backend_vars == set()

    backend_state._backend_var_private = 1
    assert backend_state.dirty_vars == set()
    assert backend_state.dirty_backend_vars == {"_backend_var_private"}
    assert backend_state.get_delta() == {
        "backend_state": {
            "backend_both": "10",
            "backend_var_private_rendered": "1",
            "computed_both": "10",
        }
    }
    backend_state.clean()


def test_set_dirty_var_backend_substate(backend_state_with_substate):
    """Test changing state backend vars updates ComputedVar in substate.

    Args:
        backend_state_with_substate: A tuple of (state, substate).
    """
    backend_state, substate = backend_state_with_substate
    # Initially there should be no dirty vars.
    assert backend_state.dirty_vars == set()
    assert backend_state.dirty_backend_vars == set()
    assert substate.dirty_vars == set()
    assert substate.dirty_backend_vars == set()

    backend_state._backend_var_private = 1
    assert substate.dirty_vars == set()
    assert substate.dirty_backend_vars == set()
    assert substate.get_full_name() not in backend_state.get_delta()
    backend_state.clean()

    backend_state._backend_var = 2
    assert substate.dirty_vars == set()
    assert substate.dirty_backend_vars == {"_backend_var"}
    assert backend_state.dirty_vars == set()
    assert backend_state.dirty_backend_vars == {"_backend_var"}
    assert backend_state.get_delta() == {
        backend_state.get_name(): {
            "backend_both": "12",
            "backend_var_rendered": "2",
            "computed_both": "12",
        },
        substate.get_full_name(): {"backend_var_processed": 6},
    }
    backend_state.clean()
    substate.clean()


def test_computed_var_recursive():
    """Check that delta is properly resolved when ComputedVar depends on ComputedVar."""

    class ComputedRecursiveState(State):
        v: int = 0
        unrelated: int = -5

        @ComputedVar
        def stage1(self) -> int:
            return self.v + 1

        @ComputedVar
        def stage2(self) -> int:
            return self.stage1 * 2

        @ComputedVar
        def stage3(self) -> int:
            return self.stage2 * self.stage2

        @ComputedVar
        def unrelated_computed(self) -> int:
            return self.unrelated

    crs = ComputedRecursiveState()
    crs.dict()
    assert crs.computed_var_dependencies == {
        "v": {"stage1"},
        "stage1": {"stage2"},
        "stage2": {"stage3"},
        "unrelated": {"unrelated_computed"},
    }
    crs.v = 1
    assert crs.get_delta() == {
        crs.get_full_name(): {"v": 1, "stage1": 2, "stage2": 4, "stage3": 16}
    }


@pytest.fixture
def interdependent_state() -> Tuple[State, List[mock.Mock]]:
    """A state with 3 computed vars with varying dependencies."""

    v1x2_mock = mock.Mock()
    v2x2_mock = mock.Mock()
    v2x2x2_mock = mock.Mock()

    class S(State):
        x: int = 0
        v1: int = 0
        v2: int = 1

        @ComputedVar
        def v1x2(self) -> int:
            v1x2_mock()
            return self.v1 * 2

        @ComputedVar
        def v2x2(self) -> int:
            v2x2_mock()
            return self.v2 * 2

        @ComputedVar
        def v2x2x2(self) -> int:
            v2x2x2_mock()
            return self.v2 * 2 * 2

    s = S()
    s.dict()  # prime initial relationships by accessing all ComputedVars
    mocks = [v1x2_mock, v2x2_mock, v2x2x2_mock]
    for m in mocks:
        m.reset_mock()
    return s, mocks


def test_set_dirty_var_no_recompute(interdependent_state):
    """Test changing non-dependent Var doesn't recalc ComputedVar."""
    state, mocks = interdependent_state
    state.x = 5
    assert state.get_delta() == {"s": {"x": 5}}
    for m in mocks:
        # expect no recalculation
        m.assert_not_called()


def test_set_dirty_var_partial_recompute_1(interdependent_state):
    """Test changing dependent Var only recalc specific ComputedVar."""
    state, (v1x2_mock, v2x2_mock, v2x2x2_mock) = interdependent_state
    state.v1 = 1
    assert state.get_delta() == {"s": {"v1": 1, "v1x2": 2}}
    v1x2_mock.assert_called()
    v2x2_mock.assert_not_called()
    v2x2x2_mock.assert_not_called()


def test_set_dirty_var_partial_recompute_2(interdependent_state):
    """Test changing dependent Var only recalc ComputedVar (and dependents thereof)."""
    state, (v1x2_mock, v2x2_mock, v2x2x2_mock) = interdependent_state
    state.v2 = 2
    assert state.get_delta() == {"s": {"v2": 2, "v2x2": 4, "v2x2x2": 8}}
    v1x2_mock.assert_not_called()
    v2x2_mock.assert_called()
    v2x2x2_mock.assert_called()


def test_set_dirty_substate(test_state, child_state, child_state2, grandchild_state):
    """Test changing substate vars marks the value as dirty.

    Args:
        test_state: A state.
        child_state: A child state.
        child_state2: A child state.
        grandchild_state: A grandchild state.
    """
    # Initially there should be no dirty vars.
    assert test_state.dirty_vars == set()
    assert child_state.dirty_vars == set()
    assert child_state2.dirty_vars == set()
    assert grandchild_state.dirty_vars == set()

    # Setting a var should mark it as dirty.
    child_state.value = "test"
    assert child_state.dirty_vars == {"value"}
    assert test_state.dirty_substates == {"child_state"}
    assert child_state.dirty_substates == set()

    # Cleaning the parent state should remove the dirty substate.
    test_state.clean()
    assert test_state.dirty_substates == set()
    assert child_state.dirty_vars == set()

    # Setting a var on the grandchild should bubble up.
    grandchild_state.value2 = "test2"
    assert child_state.dirty_substates == {"grandchild_state"}
    assert test_state.dirty_substates == {"child_state"}

    # Cleaning the middle state should keep the parent state dirty.
    child_state.clean()
    assert test_state.dirty_substates == {"child_state"}
    assert child_state.dirty_substates == set()
    assert grandchild_state.dirty_vars == set()


def test_reset(test_state, child_state):
    """Test resetting the state.

    Args:
        test_state: A state.
        child_state: A child state.
    """
    # Set some values.
    test_state.num1 = 1
    test_state.num2 = 2
    child_state.value = "test"

    # Reset the state.
    test_state.reset()

    # The values should be reset.
    assert test_state.num1 == 0
    assert test_state.num2 == 3.14
    assert child_state.value == ""

    # The dirty vars should be reset.
    assert test_state.dirty_vars == set()
    assert child_state.dirty_vars == set()

    # The dirty substates should be reset.
    assert test_state.dirty_substates == set()


@pytest.mark.asyncio
async def test_process_event_simple(test_state):
    """Test processing an event.

    Args:
        test_state: A state.
    """
    assert test_state.num1 == 0

    event = Event(token="t", name="set_num1", payload={"value": 69})
    update = await test_state.process(event)

    # The event should update the value.
    assert test_state.num1 == 69

    # The delta should contain the changes, including computed vars.
    assert update.delta == {"test_state": {"num1": 69, "sum": 72.14}}
    assert update.events == []


@pytest.mark.asyncio
async def test_process_event_substate(test_state, child_state, grandchild_state):
    """Test processing an event on a substate.

    Args:
        test_state: A state.
        child_state: A child state.
        grandchild_state: A grandchild state.
    """
    # Events should bubble down to the substate.
    assert child_state.value == ""
    assert child_state.count == 23
    event = Event(
        token="t", name="child_state.change_both", payload={"value": "hi", "count": 12}
    )
    update = await test_state.process(event)
    assert child_state.value == "HI"
    assert child_state.count == 24
    assert update.delta == {
        "test_state.child_state": {"value": "HI", "count": 24},
    }
    test_state.clean()

    # Test with the granchild state.
    assert grandchild_state.value2 == ""
    event = Event(
        token="t",
        name="child_state.grandchild_state.set_value2",
        payload={"value": "new"},
    )
    update = await test_state.process(event)
    assert grandchild_state.value2 == "new"
    assert update.delta == {
        "test_state.child_state.grandchild_state": {"value2": "new"},
    }


def test_format_event_handler():
    """Test formatting an event handler."""
    assert (
        format.format_event_handler(TestState.do_something) == "test_state.do_something"  # type: ignore
    )
    assert (
        format.format_event_handler(ChildState.change_both)  # type: ignore
        == "test_state.child_state.change_both"
    )
    assert (
        format.format_event_handler(GrandchildState.do_nothing)  # type: ignore
        == "test_state.child_state.grandchild_state.do_nothing"
    )


def test_get_token(test_state):
    assert test_state.get_token() == ""

    token = "b181904c-3953-4a79-dc18-ae9518c22f05"
    test_state.router_data = {RouteVar.CLIENT_TOKEN: token}

    assert test_state.get_token() == token


def test_get_sid(test_state):
    """Test getting session id.

    Args:
        test_state: A state.
    """
    assert test_state.get_sid() == ""

    sid = "9fpxSzPb9aFMb4wFAAAH"
    test_state.router_data = {RouteVar.SESSION_ID: sid}

    assert test_state.get_sid() == sid


def test_get_headers(test_state):
    """Test getting client headers.

    Args:
        test_state: A state.
    """
    assert test_state.get_headers() == {}

    headers = {"host": "localhost:8000", "connection": "keep-alive"}
    test_state.router_data = {RouteVar.HEADERS: headers}

    assert test_state.get_headers() == headers


def test_get_client_ip(test_state):
    """Test getting client IP.

    Args:
        test_state: A state.
    """
    assert test_state.get_client_ip() == ""

    client_ip = "127.0.0.1"
    test_state.router_data = {RouteVar.CLIENT_IP: client_ip}

    assert test_state.get_client_ip() == client_ip


def test_get_current_page(test_state):
    assert test_state.get_current_page() == ""

    route = "mypage/subpage"
    test_state.router_data = {RouteVar.PATH: route}

    assert test_state.get_current_page() == route


def test_get_query_params(test_state):
    assert test_state.get_query_params() == {}

    params = {"p1": "a", "p2": "b"}
    test_state.router_data = {RouteVar.QUERY: params}

    assert test_state.get_query_params() == params


def test_add_var(test_state):
    test_state.add_var("dynamic_int", int, 42)
    assert test_state.dynamic_int == 42

    test_state.add_var("dynamic_list", List[int], [5, 10])
    assert test_state.dynamic_list == [5, 10]
    assert test_state.dynamic_list == [5, 10]

    # how to test that one?
    # test_state.dynamic_list.append(15)
    # assert test_state.dynamic_list == [5, 10, 15]

    test_state.add_var("dynamic_dict", Dict[str, int], {"k1": 5, "k2": 10})
    assert test_state.dynamic_dict == {"k1": 5, "k2": 10}
    assert test_state.dynamic_dict == {"k1": 5, "k2": 10}
