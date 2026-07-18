from sqlalchemy import create_engine, Column, String, DateTime, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker,relationship
from datetime import datetime,timezone
import uuid
SQLALCHEMY_DATABASE_URL = "sqlite:///./researchmate.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db=SessionLocal()
    try:
        yield db
    finally:
        db.close()
class User(Base):
    __tablename__="users"
    id=Column(String,primary_key=True,default=lambda:str(uuid.uuid4()))
    email=Column(String,unique=True,nullable=False,index=True)
    password_hash=Column(String,nullable=False)
    full_name=Column(String,nullable=False)
    created_at=Column(DateTime,default=datetime.now(timezone.utc))
    team_memberships=relationship("TeamMember",back_populates="user")

class Team(Base):
    __tablename__="teams"
    id=Column(String,primary_key=True,default=lambda:str(uuid.uuid4()))
    name=Column(String,nullable=False)
    created_by = Column(String, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

    members = relationship("TeamMember", back_populates="team")
class TeamMember(Base):
    __tablename__="team_members"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    team_id = Column(String, ForeignKey("teams.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    role = Column(Enum("owner", "member", name="member_role"), default="member")
    joined_at = Column(DateTime, default=datetime.now(timezone.utc))
    team = relationship("Team", back_populates="members")
    user = relationship("User", back_populates="team_memberships")
Base.metadata.create_all(bind=engine)