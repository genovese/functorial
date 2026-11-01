# Changelog

## 0.1.5 - 2026-10-31

### Added

- Resource monad for thread-safe handling of resources
  with guarantees on release and flexible use.

- SumType structure and metaclass for defining true
  sum types using Python's class mechanism.

- SequenceView for viewing slices of a list and other
  sequences. Useful with uncons.

- performant uncons method for List

- eq and untree methods for RoseTree

- Many, many more tests

### Changed

- Improved README

- Improved setup, CI, and config

- Improved matching in LeafyBinaryTree

### Fixed

- Bug in IO that affected timing of nested IO actions

- Bug in IO that allowed some exceptions to leak from a do block

- Robustified List and NonEmptyList

- Improved type annotations in singleton.py

- Minor fixes to trees

## 0.1.4 - 2026-09-24

### Added

- Sorting utilities in List. `List.sorted` class method
  behaves like the built-in `sorted` but returns a `List`.
  The instance-method `schwartian` does a generalized
  schwarzian transform sort.

## 0.1.3 - 2026-09-23

### Added

- `map_maybe` and `cat_maybes` to List

- Several free methods to `Foldable_` including
  a version of `map_maybe`. Like `to_list`
  and `cat_maybes` in this class, these return
  only a list, rather than a List, because of
  circular import issues. Even better would
  be returning the same Foldable type, but
  this is not done yet, making these methods
  somewhat less useful without an extra
  wrapper.

- Version tags added in README

### Fixed

- Maybe's empty was a property rather than a class method.
  It has now been changed to the latter, which fixes a
  problem with e.g., guard, and is more constitent with
  its usage on *types* as well as instances.

- List's empty method had a vestigal signature
  and incorrect class access.

- Fixed a type ignore comment that got moved in a reformat.

## 0.1.2 - 2026-09-23

### Added

- ipreview optics and related tests

- new tests

- Github test actions

### Changed

- dev testing and IDE packages added to elpy deps

### Fixed

- Fixed remaining ZZZ_ classes. If ZZZ is a protocol
  representing some typeclass/trait, then ZZZ_ is
  a subclass of that that is designed to be inherited.
  This provides default implementations of a variety
  of functions in the class. Several of these ZZZ_
  classes were missing, incomplete, or inconstistent,
  which has now been fixed. This includes (Indexed)Traversable_,
  Alternative_, and Bitraversable_.

- bug on bitraversable

## 0.1.1 - 2026-09-20

Initial release. Cleaned up the structure and filled out some of the
basic holes, and added rough versions of the infrastructure, like
README.
