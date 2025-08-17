import User from "../models/user.model.js";

export const getUserSavedPosts = async (req, res) => {
  const clerkUserId = req.auth.userId;

  console.log("🔑 Clerk UserId:", clerkUserId);

  if (!clerkUserId) {
    console.log("❌ No Clerk UserId found in req.auth");
    return res.status(401).json("Not authenticated!");
  }

  const user = await User.findOne({ clerkUserId });
  console.log("🗄️ User found in DB:", user);

  if (!user) {
    return res.status(404).json("User not found!");
  }

  console.log("📌 Saved posts:", user.savedPosts);
  res.status(200).json(user.savedPosts);
};

export const savePost = async (req, res) => {
  const clerkUserId = req.auth.userId;
  const postId = req.body.postId;

  console.log("🔑 Clerk UserId:", clerkUserId);
  console.log("📝 PostId from body:", postId);

  if (!clerkUserId) {
    console.log("❌ No Clerk UserId found in req.auth");
    return res.status(401).json("Not authenticated!");
  }

  const user = await User.findOne({ clerkUserId });
  console.log("🗄️ User found in DB:", user);

  if (!user) {
    return res.status(404).json("User not found!");
  }

  const isSaved = user.savedPosts.some((p) => p.toString() === postId);
  console.log("✅ Is post already saved:", isSaved);

  if (!isSaved) {
    await User.findByIdAndUpdate(user._id, {
      $push: { savedPosts: postId },
    });
    console.log("📌 Post saved:", postId);
  } else {
    await User.findByIdAndUpdate(user._id, {
      $pull: { savedPosts: postId },
    });
    console.log("🗑️ Post unsaved:", postId);
  }

  res.status(200).json(isSaved ? "Post unsaved" : "Post saved");
};
