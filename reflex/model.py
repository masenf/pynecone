"""Database built into Reflex."""

from collections import defaultdict
from pathlib import Path
from typing import Optional

import sqlmodel

from . import constants
from reflex.base import Base
from reflex.config import get_config


def get_engine(url: Optional[str] = None):
    """Get the database engine.

    Args:
        url: the DB url to use.

    Returns:
        The database engine.

    Raises:
        ValueError: If the database url is None.
    """
    conf = get_config()
    url = url or conf.db_url
    if url is None:
        raise ValueError("No database url configured")
    return sqlmodel.create_engine(
        url,
        echo=False,
        connect_args={"check_same_thread": False} if conf.admin_dash else {},
    )


class Model(Base, sqlmodel.SQLModel):
    """Base class to define a table in the database."""

    # The primary key for the table.
    id: Optional[int] = sqlmodel.Field(primary_key=True)

    def __init_subclass__(cls):
        """Drop the default primary key field if any primary key field is defined."""
        non_default_primary_key_fields = [
            field_name
            for field_name, field in cls.__fields__.items()
            if field_name != "id" and getattr(field.field_info, "primary_key", None)
        ]
        if non_default_primary_key_fields:
            cls.__fields__.pop("id", None)

        super().__init_subclass__()

    def dict(self, **kwargs):
        """Convert the object to a dictionary.

        Args:
            kwargs: Ignored but needed for compatibility.

        Returns:
            The object as a dictionary.
        """
        return {name: getattr(self, name) for name in self.__fields__}

    @staticmethod
    def create_all():
        """Create all the tables."""
        engine = get_engine()
        sqlmodel.SQLModel.metadata.create_all(engine)

    @staticmethod
    def get_db_engine():
        """Get the database engine.

        Returns:
            The database engine.
        """
        return get_engine()

    @staticmethod
    def _alembic_render_item(type_, obj, autogen_context):
        autogen_context.imports.add("import sqlmodel")
        return False

    @classmethod
    def automigrate(cls) -> Optional[bool]:
        if not Path(constants.ALEMBIC_CONFIG).exists():
            return
        try:
            import alembic.autogenerate
            import alembic.command
            import alembic.operations.ops
            import alembic.runtime.environment
            import alembic.script
            import alembic.util
            from alembic.config import Config
        except ImportError:
            return

        config = Config(constants.ALEMBIC_CONFIG)
        script_directory = alembic.script.ScriptDirectory(config.get_main_option("script_location"))
        revision_context = alembic.autogenerate.RevisionContext(
            config=config,
            script_directory=script_directory,
            command_args=defaultdict(lambda: None, autogenerate=True, head="head"),
        )
        writer = alembic.autogenerate.rewriter.Rewriter()

        def run_autogenerate(rev, context):
            revision_context.run_autogenerate(rev, context)
            return []

        def run_upgrade(rev, context):
            return script_directory._upgrade_revs("head", rev)

        @writer.rewrites(alembic.operations.ops.AddColumnOp)
        def render_add_column_with_default(context, revision, op):
            if op.column.nullable:
                return op
            else:
                op.column.nullable = True
                return [
                    op,
                    alembic.operations.ops.ExecuteSQLOp(
                        f"UPDATE {op.table_name} SET {op.column.name} = {op.column.default.arg!r}"
                    ),
                    alembic.operations.ops.AlterColumnOp(
                        op.table_name,
                        op.column.name,
                        modify_nullable=False,
                        existing_type=op.column.type,
                    ),
                ]

        with cls.get_db_engine().connect() as connection:
            with alembic.runtime.environment.EnvironmentContext(
                config=config,
                script=script_directory,
                fn=run_autogenerate,
            ) as env:
                env.configure(
                    connection=connection,
                    target_metadata=sqlmodel.SQLModel.metadata,
                    render_item=cls._alembic_render_item,
                    process_revision_directives=writer,
                )
                env.run_migrations()
            if revision_context.generated_revisions[-1].upgrade_ops.ops:
                for s in revision_context.generate_scripts():
                    pass
                # apply updates to database
                with alembic.runtime.environment.EnvironmentContext(
                    config=config,
                    script=script_directory,
                    fn=run_upgrade,
                ) as env:
                    env.configure(connection=connection)
                    env.run_migrations()
                    connection.commit()
        return True

    @classmethod
    @property
    def select(cls):
        """Select rows from the table.

        Returns:
            The select statement.
        """
        return sqlmodel.select(cls)


def session(url: Optional[str] = None) -> sqlmodel.Session:
    """Get a session to interact with the database.

    Args:
        url: The database url.

    Returns:
        A database session.
    """
    return sqlmodel.Session(get_engine(url))
